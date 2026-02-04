"""
EmbSpatial-Bench Dataset Implementation

EmbSpatial-Bench is a spatial understanding benchmark.
- Dataset: HuggingFace `FlagEval/EmbSpatial-Bench`
- Split: "test"
- Format: MCQ with up to 6 options
- Evaluation: Exact match of answer letter
"""

import os
import pandas as pd
from datasets import load_dataset
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter, index_to_letter, build_mcq_prompt
from . import EMBODIED_DATA_ROOT


class EmbSpatialBenchDataset(ImageBaseDataset):
    """EmbSpatial-Bench: Embodied Spatial Understanding Benchmark."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['EmbSpatial_Embodied']

    def __init__(self, dataset='EmbSpatial_Embodied', **kwargs):
        self.dataset_name = dataset
        # Directory to save images
        self.img_root = os.path.join(EMBODIED_DATA_ROOT, 'emb_spatial', 'images')
        os.makedirs(self.img_root, exist_ok=True)
        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace and save images to disk."""
        dataset_path = "FlagEval/EmbSpatial-Bench"

        try:
            hf_dataset = load_dataset(dataset_path, split="test")
        except Exception:
            hf_dataset = load_dataset(dataset_path, split="train")

        # Convert to DataFrame format
        data_list = []
        for idx, item in enumerate(hf_dataset):
            # Convert GT index to letter
            gt_index = item['answer']
            gt_letter = index_to_letter(gt_index)

            # Save image to disk (models expect file paths, not PIL objects)
            image_path = os.path.join(self.img_root, f'{idx:06d}.jpg')
            if not os.path.exists(image_path):
                item['image'].save(image_path)

            data_list.append({
                'index': idx,
                'image_path': image_path,
                'question': item['question'],
                'options': item['answer_options'],  # List of options
                'answer_index': gt_index,
                'answer': gt_letter,  # "(A)", "(B)", etc.
            })

        self.data = pd.DataFrame(data_list)
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'image_path': item['image_path'],
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build MCQ prompt."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']
        options = line['options']

        # Build MCQ prompt
        prompt = build_mcq_prompt(question, options)

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=prompt),
        ]
        return msgs

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [line['image_path']]

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

            # Extract answer letter (supports up to F)
            pred_letter = extract_answer_letter(pred_text, max_letter='F')

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
