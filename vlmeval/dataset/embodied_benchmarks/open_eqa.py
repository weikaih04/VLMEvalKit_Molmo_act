"""
OpenEQA Dataset Implementation

OpenEQA is an open-ended video QA benchmark.
- Dataset: JSON metadata + local frame folders
- Format: Video VQA (open-ended)
- Evaluation: Only saves predictions (no automatic scoring)

Frames are stored as individual images in folders:
  {data_root}/frames/{episode_history}/00001.jpg, 00002.jpg, ...
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
from ..video_base import VideoBaseDataset
from ...smp import load, dump
from . import EMBODIED_DATA_ROOT


def get_frames_from_folder(folder_path, num_frames=8):
    """Load frames from a folder of images."""
    if not os.path.exists(folder_path):
        return []

    # Find all images
    files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )

    if not files:
        return []

    # Uniformly sample indices
    indices = np.linspace(0, len(files) - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        try:
            img = Image.open(files[idx]).convert("RGB")
            frames.append(img)
        except Exception:
            pass

    return frames


class OpenEQADataset(VideoBaseDataset):
    """OpenEQA: Open-ended Embodied Question Answering.

    Note: This benchmark only saves predictions. No automatic evaluation is performed.
    """

    TYPE = 'VQA'
    MODALITY = 'VIDEO'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['OpenEQA_Embodied', 'OpenEQA']

    def __init__(self, dataset='OpenEQA_Embodied', **kwargs):
        self.dataset_name = dataset
        self.num_frames = kwargs.get('num_frames', 8)

        # Data root containing frames folder
        self.data_root = kwargs.get('data_root', os.path.join(EMBODIED_DATA_ROOT, 'openeqa'))

        # JSON metadata path
        self.json_path = kwargs.get('json_path', os.path.join(self.data_root, 'open-eqa-v0.json'))

        self._load_local_dataset()

    def _load_local_dataset(self):
        """Load dataset from local JSON and frame folders."""
        if not os.path.exists(self.json_path):
            print(f"Warning: OpenEQA JSON not found at {self.json_path}")
            self.data = pd.DataFrame()
            return

        with open(self.json_path, 'r') as f:
            json_data = json.load(f)

        data_list = []
        idx_counter = 0

        for item in json_data:
            question = item['question']
            gt_answer = item['answer']
            rel_path = item['episode_history']  # e.g. "hm3d-v0/000-hm3d-BFRyYbPCCPE"

            # Construct full path
            full_path = os.path.join(self.data_root, "frames", rel_path)

            # Only include samples with valid frame folders
            if not os.path.exists(full_path):
                continue

            data_list.append({
                'index': idx_counter,
                'episode_history': rel_path,
                'frames_path': full_path,
                'question': question,
                'answer': gt_answer,
            })
            idx_counter += 1

        self.data = pd.DataFrame(data_list)

        if len(self.data) > 0:
            print(f"OpenEQA: Loaded {len(self.data)} samples with valid frame folders")
        else:
            print(f"Warning: No valid frame folders found in {os.path.join(self.data_root, 'frames')}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'episode_history': item['episode_history'],
            'frames_path': item['frames_path'],
            'question': item['question'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build prompt with video frames."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        frames_path = line['frames_path']
        question = line['question']

        # Load frames
        frames = get_frames_from_folder(frames_path, self.num_frames)

        # Build message: frames + question
        msgs = []
        for frame in frames:
            msgs.append(dict(type='image', value=frame))
        msgs.append(dict(type='text', value=question))

        return msgs

    def dump_image(self, line):
        """Return video frames."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        frames_path = line['frames_path']
        return get_frames_from_folder(frames_path, self.num_frames)

    def evaluate(self, eval_file, **judge_kwargs):
        """Save predictions only - no automatic evaluation.

        OpenEQA is an open-ended QA benchmark that requires human evaluation
        or LLM-as-judge for proper scoring.

        Returns:
            dict with prediction/answer pairs saved to result file
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))
            episode = row.get('episode_history', '')
            question = row.get('question', '')

            results.append({
                'index': row.get('index', idx),
                'episode_history': episode,
                'question': question,
                'prediction': pred_text,
                'answer': gt_answer,
            })

        # Save results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        # Return info about saved predictions
        return {
            'total': len(results),
            'note': 'OpenEQA predictions saved. Manual evaluation required.',
            'result_file': result_file,
        }
