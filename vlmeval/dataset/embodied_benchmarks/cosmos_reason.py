"""
Cosmos-Reason 1 Dataset Implementation

Cosmos-Reason 1 is a video reasoning benchmark.
- Dataset: Local JSON files `{subset}_benchmark_qa_pairs.json`
- Subsets: agibot, bridgev2, holoassist, robofail, robovqa
- Format: Video MCQ
- Evaluation: Exact match of answer letter
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from ..video_base import VideoBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter
from . import EMBODIED_DATA_ROOT


# Known subsets (only robofail and robovqa as requested)
COSMOS_SUBSETS = ["robofail", "robovqa"]


def extract_frames(video_path, num_frames=8, frame_cache_dir=None):
    """Extract uniformly spaced frames from video and save to disk.

    Uses decord if available (faster), falls back to OpenCV.
    Returns list of file paths instead of PIL Image objects.
    """
    if not os.path.exists(video_path):
        return []

    # Create frame cache directory if provided
    if frame_cache_dir is None:
        frame_cache_dir = os.path.join(EMBODIED_DATA_ROOT, 'cosmos_reason1', 'frames')
    os.makedirs(frame_cache_dir, exist_ok=True)

    # Generate unique ID for this video
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(video_path))
    frame_prefix = f"{parent_dir}_{video_id}"

    # Check if frames already cached
    frame_paths = [os.path.join(frame_cache_dir, f"{frame_prefix}_{i:03d}.jpg") for i in range(num_frames)]
    if all(os.path.exists(p) for p in frame_paths):
        return frame_paths

    # Try decord first (much faster for video frame extraction)
    frames = []
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_array = vr.get_batch(indices).asnumpy()  # Batch extraction is faster
        frames = [Image.fromarray(frame) for frame in frames_array]
    except ImportError:
        pass  # Fall back to OpenCV
    except Exception:
        pass  # Fall back to OpenCV on any error

    # OpenCV fallback
    if not frames:
        try:
            import cv2
        except ImportError:
            print("Warning: Neither decord nor OpenCV available for video loading")
            return []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()

    # Save frames to disk and return paths
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frame_cache_dir, f"{frame_prefix}_{i:03d}.jpg")
        if not os.path.exists(frame_path):
            frame.convert('RGB').save(frame_path, 'JPEG')
        frame_paths.append(frame_path)

    return frame_paths


def format_cosmos_prompt(qa_data):
    """Construct prompt from QA data."""
    question = qa_data.get("question", "Which description best matches the video?")

    options_dict = qa_data.get("index2ans", {})
    # Sort keys to ensure A, B, C, D order
    sorted_keys = sorted(options_dict.keys())

    formatted_options = []
    for key in sorted_keys:
        formatted_options.append(f"({key}) {options_dict[key]}")

    prompt = f"{question}\n" + "\n".join(formatted_options) + "\nAnswer with the option letter."
    return prompt


class CosmosReason1Dataset(VideoBaseDataset):
    """Cosmos-Reason 1: Video Reasoning Benchmark."""

    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        datasets = ['CosmosReason1_Embodied']
        for subset in COSMOS_SUBSETS:
            datasets.append(f'CosmosReason1_{subset}')
        return datasets

    def __init__(self, dataset='CosmosReason1_Embodied', **kwargs):
        self.dataset_name = dataset
        self.num_frames = kwargs.get('num_frames', 8)

        # Data root
        self.data_root = kwargs.get('data_root', os.path.join(EMBODIED_DATA_ROOT, 'cosmos_reason1'))

        # Determine which subsets to load
        if dataset == 'CosmosReason1_Embodied':
            self.subsets = COSMOS_SUBSETS
        else:
            subset_name = dataset.replace('CosmosReason1_', '')
            if subset_name in COSMOS_SUBSETS:
                self.subsets = [subset_name]
            else:
                self.subsets = COSMOS_SUBSETS

        self._load_local_dataset()

    def _load_local_dataset(self):
        """Load dataset from local JSON files."""
        data_list = []
        idx_counter = 0

        for subset in self.subsets:
            subset_dir = os.path.join(self.data_root, subset)
            json_path = os.path.join(subset_dir, f"{subset}_benchmark_qa_pairs.json")

            if not os.path.exists(json_path):
                print(f"Warning: Skipping {subset}: JSON not found at {json_path}")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Flatten the data structure
            for item in data:
                video_rel_path = item.get("video")  # e.g. "clips/..."
                qa_content = item.get("qa_pairs")

                # qa_content might be a dict or list of dicts
                if isinstance(qa_content, dict):
                    qa_list = [qa_content]
                elif isinstance(qa_content, list):
                    qa_list = qa_content
                else:
                    continue

                for qa in qa_list:
                    video_path = os.path.join(subset_dir, video_rel_path) if video_rel_path else None
                    gt_answer = qa.get("answer") or qa.get("correct_answer")

                    data_list.append({
                        'index': idx_counter,
                        'subset': subset,
                        'video_path': video_path,
                        'qa_data': qa,
                        'answer': gt_answer,
                    })
                    idx_counter += 1

        self.data = pd.DataFrame(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'subset': item['subset'],
            'video_path': item['video_path'],
            'qa_data': item['qa_data'],
            'answer': item['answer'],
        }

    def build_prompt(self, line, video_llm=False):
        """Build prompt with video or image frames.

        Args:
            line: Data row or index
            video_llm: If True and video file exists, use type='video'.
                      Otherwise, use type='image' for each frame.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_path = line['video_path']
        qa_data = line['qa_data']

        # Build prompt text
        prompt_text = format_cosmos_prompt(qa_data)

        msgs = []

        # Use video mode if video_llm=True and video file exists
        if video_llm and video_path and os.path.exists(video_path):
            msgs.append(dict(type='video', value=video_path))
        else:
            # Fallback to multi-image mode (extract frames)
            frame_paths = []
            if video_path and os.path.exists(video_path):
                frame_paths = extract_frames(video_path, self.num_frames)
            for frame_path in frame_paths:
                msgs.append(dict(type='image', value=frame_path))

        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def dump_image(self, line):
        """Return video frame paths."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_path = line['video_path']
        if video_path and os.path.exists(video_path):
            return extract_frames(video_path, self.num_frames)
        return []

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        Cosmos-Reason 1 evaluation: exact match of answer letter.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        # Track per-subset accuracy
        subset_correct = {s: 0 for s in COSMOS_SUBSETS}
        subset_total = {s: 0 for s in COSMOS_SUBSETS}

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))
            subset = row.get('subset', 'unknown')

            # Extract answer letter
            pred_letter = extract_answer_letter(pred_text, max_letter='E')

            is_correct = pred_letter is not None and pred_letter == gt_answer

            if subset in subset_correct:
                if is_correct:
                    subset_correct[subset] += 1
                subset_total[subset] += 1

            results.append({
                'index': row.get('index', idx),
                'subset': subset,
                'prediction': pred_text,
                'parsed_prediction': pred_letter,
                'answer': gt_answer,
                'correct': is_correct,
            })

        # Calculate per-subset accuracy
        subset_accuracy = {}
        for subset in COSMOS_SUBSETS:
            if subset_total[subset] > 0:
                subset_accuracy[subset] = subset_correct[subset] / subset_total[subset] * 100
            else:
                subset_accuracy[subset] = 0.0

        # Calculate average accuracy (across subsets)
        valid_subsets = [s for s in COSMOS_SUBSETS if subset_total[s] > 0]
        if valid_subsets:
            average_accuracy = sum(subset_accuracy[s] for s in valid_subsets) / len(valid_subsets)
        else:
            average_accuracy = 0.0

        # Save detailed results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        # Return summary
        summary = {
            'average_accuracy': average_accuracy,
        }
        for subset in valid_subsets:
            summary[f'{subset}_accuracy'] = subset_accuracy[subset]

        return summary
