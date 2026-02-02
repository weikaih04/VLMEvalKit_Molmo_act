"""
RefSpatial-Bench Dataset Implementation

RefSpatial-Bench is a pointing benchmark for spatial referring.
- Dataset: HuggingFace `BAAI/RefSpatial-Bench`
- Splits: "location", "placement"
- Format: Point to object, check if point is in mask
- Evaluation: Accuracy (point in mask)
"""

import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_molmo_points, check_point_in_mask


class RefSpatialBenchDataset(ImageBaseDataset):
    """RefSpatial-Bench: Spatial Referring Pointing Benchmark."""

    TYPE = 'VQA'  # Pointing task
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return [
            'RefSpatial_Embodied',
            'RefSpatial_location',
            'RefSpatial_placement',
        ]

    def __init__(self, dataset='RefSpatial_Embodied', **kwargs):
        self.dataset_name = dataset

        # Determine splits to load
        if 'location' in dataset.lower():
            self.splits = ['location']
        elif 'placement' in dataset.lower():
            self.splits = ['placement']
        else:
            self.splits = ['location', 'placement']

        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        dataset_path = "BAAI/RefSpatial-Bench"

        data_list = []
        idx_counter = 0

        for split in self.splits:
            try:
                hf_dataset = load_dataset(dataset_path, split=split)
            except Exception as e:
                print(f"Warning: Could not load RefSpatial split {split}: {e}")
                continue

            for item in hf_dataset:
                data_list.append({
                    'index': idx_counter,
                    'split': split,
                    'image': item['image'],
                    'mask': item['mask'],
                    'object': item['object'],
                    'image_width': item['image'].width,
                    'image_height': item['image'].height,
                })
                idx_counter += 1

        self.data = pd.DataFrame(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'split': item['split'],
            'image': item['image'],
            'mask': item['mask'],
            'object': item['object'],
        }

    def build_prompt(self, line):
        """Build pointing prompt."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image = line['image']
        obj = line['object']

        # Standard pointing prompt
        prompt = f"Point to the {obj}."

        msgs = [
            dict(type='image', value=image),
            dict(type='text', value=prompt),
        ]
        return msgs

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['image']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate pointing predictions.

        RefSpatial evaluation: check if predicted point is in ground truth mask.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        # Track per-split accuracy
        split_correct = {'location': 0, 'placement': 0}
        split_total = {'location': 0, 'placement': 0}

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            split = row.get('split', 'unknown')

            # Get mask and image dimensions from original data
            orig_idx = row.get('index', idx)
            if orig_idx < len(self.data):
                orig_row = self.data.iloc[orig_idx]
                mask = orig_row['mask']
                img_width = orig_row['image_width']
                img_height = orig_row['image_height']
            else:
                # Fallback if index doesn't match
                results.append({
                    'index': orig_idx,
                    'split': split,
                    'prediction': pred_text,
                    'correct': False,
                })
                continue

            # Extract points from prediction
            points = extract_molmo_points(pred_text)

            is_hit = False
            pred_point = None
            if points:
                pred_point = points[0]  # Take first point
                is_hit = check_point_in_mask(
                    pred_point[0], pred_point[1],
                    mask, img_width, img_height
                )

            if split in split_correct:
                if is_hit:
                    split_correct[split] += 1
                split_total[split] += 1

            results.append({
                'index': orig_idx,
                'split': split,
                'prediction': pred_text,
                'parsed_point': pred_point,
                'correct': is_hit,
            })

        # Calculate per-split accuracy
        split_accuracy = {}
        for split in ['location', 'placement']:
            if split_total[split] > 0:
                split_accuracy[split] = split_correct[split] / split_total[split] * 100
            else:
                split_accuracy[split] = 0.0

        # Calculate overall accuracy
        total_correct = sum(split_correct.values())
        total_samples = sum(split_total.values())
        overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0

        # Save detailed results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        return {
            'accuracy': overall_accuracy,
            'location_accuracy': split_accuracy['location'],
            'placement_accuracy': split_accuracy['placement'],
            'correct': total_correct,
            'total': total_samples,
        }
