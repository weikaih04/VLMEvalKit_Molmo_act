"""
Minimal Video Pairs Dataset Implementation

Minimal Video Pairs is a video reasoning benchmark.
- Dataset: HuggingFace `facebook/minimal_video_pairs`
- Subsets: human_object_interactions, intuitive_physics, robot_object_interactions, temporal_reasoning
- Format: Video MCQ
- Evaluation: Exact match of answer letter

Note: Videos need to be downloaded separately and organized by subset.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from ..video_base import VideoBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter, index_to_letter
from . import EMBODIED_DATA_ROOT


# Known subsets
MVP_SUBSETS = [
    "human_object_interactions",
    "intuitive_physics",
    "robot_object_interactions",
    "temporal_reasoning"
]


def extract_frames(video_path, num_frames=8):
    """Extract uniformly spaced frames from video."""
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not available for video loading")
        return []

    if not video_path or not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


class MinimalVideosDataset(VideoBaseDataset):
    """Minimal Video Pairs: Video Reasoning Benchmark."""

    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        datasets = ['MinimalVideos_Embodied']
        for subset in MVP_SUBSETS:
            datasets.append(f'MinimalVideos_{subset}')
        return datasets

    def __init__(self, dataset='MinimalVideos_Embodied', split='mini', **kwargs):
        self.dataset_name = dataset
        self.split = split
        self.num_frames = kwargs.get('num_frames', 8)

        # Video data root
        self.video_root = kwargs.get('video_root', os.path.join(EMBODIED_DATA_ROOT, 'minimal_videos'))

        # Determine which subsets to load
        if dataset == 'MinimalVideos_Embodied':
            self.subsets = MVP_SUBSETS
        else:
            subset_name = dataset.replace('MinimalVideos_', '')
            if subset_name in MVP_SUBSETS:
                self.subsets = [subset_name]
            else:
                self.subsets = MVP_SUBSETS

        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load metadata from HuggingFace."""
        dataset_path = "facebook/minimal_video_pairs"

        data_list = []
        idx_counter = 0

        for subset in self.subsets:
            try:
                hf_dataset = load_dataset(dataset_path, subset, split=self.split)
            except Exception as e:
                print(f"Warning: Could not load MinimalVideos subset {subset}: {e}")
                continue

            for item in hf_dataset:
                question = item['question']
                candidates = item['candidates']  # List of strings
                gt_text = str(item['answer'])
                video_rel_path = item['video_path']

                # Find GT letter by matching answer text to candidates
                gt_letter = None
                try:
                    gt_idx = candidates.index(gt_text)
                    gt_letter = index_to_letter(gt_idx)
                except ValueError:
                    # Answer not in candidates - skip
                    continue

                # Resolve video path
                video_path = os.path.join(self.video_root, video_rel_path) if self.video_root else video_rel_path

                data_list.append({
                    'index': idx_counter,
                    'subset': subset,
                    'video_path': video_path,
                    'video_rel_path': video_rel_path,
                    'question': question,
                    'candidates': candidates,
                    'answer_text': gt_text,
                    'answer': gt_letter,  # Letter format
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
            'question': item['question'],
            'candidates': item['candidates'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build prompt with video frames."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_path = line['video_path']
        question = line['question']
        candidates = line['candidates']

        # Load video frames
        frames = []
        if video_path and os.path.exists(video_path):
            frames = extract_frames(video_path, self.num_frames)

        # Build prompt
        letters = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        formatted_options = []
        for i, opt in enumerate(candidates):
            if i < len(letters):
                formatted_options.append(f"{letters[i]} {opt}")

        prompt_text = f"{question}\n" + "\n".join(formatted_options) + "\nAnswer with the option letter."

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
            return extract_frames(video_path, self.num_frames)
        return []

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        Minimal Videos evaluation: exact match of answer letter.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        # Track per-subset accuracy
        subset_correct = {s: 0 for s in MVP_SUBSETS}
        subset_total = {s: 0 for s in MVP_SUBSETS}

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
        for subset in MVP_SUBSETS:
            if subset_total[subset] > 0:
                subset_accuracy[subset] = subset_correct[subset] / subset_total[subset] * 100
            else:
                subset_accuracy[subset] = 0.0

        # Calculate average accuracy (across subsets)
        valid_subsets = [s for s in MVP_SUBSETS if subset_total[s] > 0]
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
