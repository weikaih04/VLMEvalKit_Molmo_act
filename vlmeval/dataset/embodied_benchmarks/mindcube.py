"""
MindCube Dataset Implementation

MindCube is a spatial reasoning benchmark with multiple images.
- Dataset: Local JSONL file
- Format: MCQ with 4 images per sample
- Evaluation: Exact match of answer letter
"""

import os
import json
import pandas as pd
from PIL import Image
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter
from . import EMBODIED_DATA_ROOT


class MindCubeDataset(ImageBaseDataset):
    """MindCube: Multi-image Spatial Reasoning Benchmark."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['MindCube_Embodied', 'MindCube_tinybench']

    def __init__(self, dataset='MindCube_Embodied', **kwargs):
        self.dataset_name = dataset

        # Determine split from dataset name
        if 'tinybench' in dataset.lower():
            self.split = 'tinybench'
        else:
            self.split = 'tinybench'  # default

        self._load_local_dataset()

    def _load_local_dataset(self):
        """Load dataset from local JSONL file."""
        data_dir = os.path.join(EMBODIED_DATA_ROOT, 'mindcube', 'data')
        jsonl_path = os.path.join(data_dir, 'raw', f'MindCube_{self.split}.jsonl')

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"MindCube JSONL not found at: {jsonl_path}")

        # Load JSONL
        with open(jsonl_path, 'r') as f:
            samples = [json.loads(line) for line in f]

        data_list = []
        for idx, item in enumerate(samples):
            question = item['question']
            gt_answer = item['gt_answer']  # Single letter like "C"
            img_paths = item['images']  # List of relative paths

            # Load images
            images = self._load_images(data_dir, img_paths)
            if not images:
                continue

            data_list.append({
                'index': idx,
                'images': images,
                'image_paths': img_paths,
                'question': question,
                'answer': gt_answer,  # Single letter
            })

        self.data = pd.DataFrame(data_list)
        self.data_dir = data_dir

    def _load_images(self, data_dir, image_paths):
        """Load images from relative paths."""
        images = []
        for path in image_paths:
            full_path = os.path.join(data_dir, path)
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path).convert("RGB")
                    images.append(img)
                except Exception:
                    pass
        return images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'images': item['images'],
            'question': item['question'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build prompt with multiple images."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        images = line['images']
        question = line['question']

        # MindCube questions usually include options text
        prompt = f"{question}\nAnswer with the option letter."

        # Build message: [IMG, IMG, IMG, IMG, TEXT]
        msgs = []
        for img in images:
            msgs.append(dict(type='image', value=img))
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['images']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions."""
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))

            # Extract letter (MindCube uses single letter, not parenthesized)
            pred_letter = extract_answer_letter(pred_text, max_letter='D')
            if pred_letter:
                # Remove parentheses for comparison
                pred_letter = pred_letter.strip('()')

            is_correct = pred_letter is not None and pred_letter == gt_answer

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'index': row.get('index', idx),
                'prediction': pred_text,
                'parsed_prediction': pred_letter,
                'answer': gt_answer,
                'correct': is_correct,
            })

        accuracy = correct / total * 100 if total > 0 else 0

        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
