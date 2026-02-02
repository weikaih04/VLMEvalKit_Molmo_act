"""
VSI-Bench Dataset Implementation

VSI-Bench is a video spatial understanding benchmark.
- Dataset: HuggingFace `nyu-visionx/VSI-Bench`
- Format: Video MCQ / Counting / VQA
- Evaluation: Substring match, number extraction

Video files need to be downloaded separately and organized by dataset/scene_name.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from ..video_base import VideoBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter, normalize_text, extract_number, build_mcq_prompt
from . import EMBODIED_DATA_ROOT


def find_visual_file(data_root, dataset_name, scene_name):
    """Find video/image file by dataset and scene name."""
    extensions = ['.mp4', '.mov', '.avi', '.png', '.jpg', '.jpeg']
    dataset_dir = os.path.join(data_root, dataset_name)

    if not os.path.exists(dataset_dir):
        if os.path.basename(data_root) == dataset_name:
            dataset_dir = data_root
        else:
            return None

    # Try direct path
    for ext in extensions:
        path = os.path.join(dataset_dir, f"{scene_name}{ext}")
        if os.path.exists(path):
            return path
        # Try images subfolder
        path_img = os.path.join(dataset_dir, "images", f"{scene_name}{ext}")
        if os.path.exists(path_img):
            return path_img

    # Recursive search
    search_pattern = os.path.join(dataset_dir, "**", f"{scene_name}.*")
    matches = glob.glob(search_pattern, recursive=True)
    for m in matches:
        if m.lower().endswith(tuple(extensions)):
            return m

    return None


def load_video_frames(file_path, target_frames=16):
    """Load video frames with dynamic sampling."""
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not available for video loading")
        return []

    if file_path.lower().endswith(('.mp4', '.mov', '.avi')):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        # Calculate sampling rate to fit target_frames
        step = max(1, total_frames // target_frames)

        frames = []
        indices = np.arange(0, total_frames, step).astype(int)
        indices = indices[:target_frames]  # Cap at target

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames
    else:
        # Single image
        try:
            return [Image.open(file_path).convert("RGB")]
        except Exception:
            return []


class VSIBenchDataset(VideoBaseDataset):
    """VSI-Bench: Video Spatial Understanding Benchmark."""

    TYPE = 'VQA'
    MODALITY = 'VIDEO'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['VSI_Bench_Embodied', 'VSI_Bench']

    def __init__(self, dataset='VSI_Bench_Embodied', **kwargs):
        self.dataset_name = dataset
        self.target_frames = kwargs.get('target_frames', 16)

        # Video data root (where dataset/scene videos are stored)
        self.video_root = kwargs.get('video_root', os.path.join(EMBODIED_DATA_ROOT, 'vsi_bench'))

        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load metadata from HuggingFace."""
        dataset_path = "nyu-visionx/VSI-Bench"

        try:
            hf_dataset = load_dataset(dataset_path, split="test")
        except Exception:
            try:
                hf_dataset = load_dataset(dataset_path, split="train")
            except Exception as e:
                print(f"Warning: Could not load VSI-Bench: {e}")
                self.data = pd.DataFrame()
                return

        data_list = []
        for idx, item in enumerate(hf_dataset):
            question = item['question']
            gt_answer = str(item['ground_truth'])
            q_type = item.get('question_type', 'general')
            ds_name = item.get('dataset', 'unknown')
            scene_name = item.get('scene_name', 'unknown')
            options = item.get('options')

            # Find video file
            video_path = find_visual_file(self.video_root, ds_name, scene_name)

            data_list.append({
                'index': idx,
                'dataset': ds_name,
                'scene_name': scene_name,
                'video_path': video_path,
                'question': question,
                'question_type': q_type,
                'options': options,
                'answer': gt_answer,
            })

        self.data = pd.DataFrame(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'dataset': item['dataset'],
            'scene_name': item['scene_name'],
            'video_path': item['video_path'],
            'question': item['question'],
            'question_type': item['question_type'],
            'options': item['options'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build prompt with video frames."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_path = line['video_path']
        question = line['question']
        options = line['options']

        # Load video frames
        frames = []
        if video_path and os.path.exists(video_path):
            frames = load_video_frames(video_path, self.target_frames)

        # Build prompt
        if options:
            # MCQ format
            letters = ["(A)", "(B)", "(C)", "(D)"]
            opt_str = "\n".join([f"{letters[j]} {opt}" for j, opt in enumerate(options) if j < 4])
            prompt_text = f"{question}\n{opt_str}\nAnswer with the option letter."
        else:
            prompt_text = question

        # Build message
        msgs = []
        for frame in frames:
            msgs.append(dict(type='image', value=frame))
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def dump_image(self, line):
        """Return video frames."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_path = line['video_path']
        if video_path and os.path.exists(video_path):
            return load_video_frames(video_path, self.target_frames)
        return []

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        VSI-Bench evaluation:
        - MCQ: substring match (gt_answer in prediction)
        - Counting: number extraction and comparison
        - General: normalized substring match
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

        # Track per-type accuracy
        type_correct = {}
        type_total = {}

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))
            q_type = row.get('question_type', 'general')
            options = row.get('options')
            question = row.get('question', '')

            # Scoring based on question type
            is_correct = False

            if options:
                # MCQ: substring match
                if gt_answer.lower() in pred_text.lower():
                    is_correct = True
            elif 'count' in str(q_type).lower() or 'count' in question.lower():
                # Counting: extract number
                pred_num = extract_number(pred_text)
                if pred_num == gt_answer:
                    is_correct = True
            else:
                # General: normalized substring match
                if normalize_text(gt_answer) in normalize_text(pred_text):
                    is_correct = True

            if is_correct:
                correct += 1
            total += 1

            # Track per-type
            if q_type not in type_correct:
                type_correct[q_type] = 0
                type_total[q_type] = 0
            if is_correct:
                type_correct[q_type] += 1
            type_total[q_type] += 1

            results.append({
                'index': row.get('index', idx),
                'question_type': q_type,
                'prediction': pred_text,
                'answer': gt_answer,
                'correct': is_correct,
            })

        # Calculate overall accuracy
        accuracy = correct / total * 100 if total > 0 else 0

        # Calculate per-type accuracy
        type_accuracy = {}
        for q_type in type_correct:
            if type_total[q_type] > 0:
                type_accuracy[q_type] = type_correct[q_type] / type_total[q_type] * 100

        # Save detailed results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        # Return summary
        summary = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        # Add per-type accuracy
        for q_type, acc in type_accuracy.items():
            summary[f'{q_type}_accuracy'] = acc

        return summary
