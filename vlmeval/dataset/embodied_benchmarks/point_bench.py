"""
Point-Bench Dataset Implementation

Point-Bench is a pointing benchmark for evaluating multimodal grounding.
- Dataset: GitHub `pointarena/pointarena` or HuggingFace `PointArena/pointarena-data`
- Format: Image + Query → Point coordinates
- Categories: affordable, counting, spatial, reasoning, steerable
- Evaluation: Point-in-mask accuracy

Reference: https://pointarena.github.io/
Paper: arXiv:2505.09990
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_points_robust, extract_molmo_points
from . import EMBODIED_DATA_ROOT


def check_point_in_mask(x_pct, y_pct, mask, width, height, threshold=127):
    """Check if a point (in percentage 0-100) falls within the mask.

    Args:
        x_pct, y_pct: Point coordinates as percentage (0-100)
        mask: PIL Image or numpy array (binary mask)
        width, height: Image dimensions
        threshold: Mask value threshold (default 127 for binary masks)

    Returns:
        bool: True if point is in mask
    """
    # Convert percentage to pixel coordinates
    x_pixel = int(x_pct / 100 * width)
    y_pixel = int(y_pct / 100 * height)

    # Clamp to valid range
    x_pixel = max(0, min(x_pixel, width - 1))
    y_pixel = max(0, min(y_pixel, height - 1))

    # Convert mask to numpy if needed
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask.convert('L'))
    else:
        mask_array = np.array(mask)

    # Check if point is in mask (white pixels = target region)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel

    return mask_array[y_pixel, x_pixel] > threshold


class PointBenchDataset(ImageBaseDataset):
    """Point-Bench: Multimodal Pointing Benchmark."""

    TYPE = 'Pointing'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    # Task categories
    CATEGORIES = ['affordable', 'counting', 'spatial', 'reasoning', 'steerable']

    @classmethod
    def supported_datasets(cls):
        return ['PointBench_Embodied', 'PointBench']

    def __init__(self, dataset='PointBench_Embodied', **kwargs):
        self.dataset_name = dataset

        # Data root where Point-Bench data is stored (pointarena repo with data)
        self.data_root = kwargs.get(
            'data_root',
            os.path.join(EMBODIED_DATA_ROOT, 'pointarena')
        )

        self._load_dataset()

    def _load_dataset(self):
        """Load Point-Bench dataset from local files."""
        data_file = os.path.join(self.data_root, 'data.json')

        if not os.path.exists(data_file):
            print(f"Warning: Point-Bench data.json not found at {data_file}")
            print("Please download from: https://huggingface.co/datasets/PointArena/pointarena-data")
            self.data = pd.DataFrame()
            return

        with open(data_file, 'r') as f:
            annotations = json.load(f)

        data_list = []
        for idx, item in enumerate(annotations):
            image_filename = item.get('image_filename', '')
            mask_filename = item.get('mask_filename', '')
            user_input = item.get('user_input', '')
            category = item.get('category', 'unknown')
            count = item.get('count', 1)

            # Build full paths - images and masks are organized by category
            # Structure: selected_images/{category}/{filename}
            #            selected_masks/{category}/{filename}
            image_path = os.path.join(self.data_root, 'selected_images', category, image_filename)
            mask_path = os.path.join(self.data_root, 'selected_masks', category, mask_filename)

            # Check if files exist
            if not os.path.exists(image_path):
                # Try legacy structure: images/{category}/{filename}
                image_path = os.path.join(self.data_root, 'images', category, image_filename)

            if not os.path.exists(image_path):
                continue

            if not os.path.exists(mask_path):
                # Try legacy structure: masks/{filename}
                mask_path = os.path.join(self.data_root, 'masks', mask_filename)

            data_list.append({
                'index': idx,
                'image_path': image_path,
                'mask_path': mask_path,
                'question': user_input,
                'category': category,
                'count': count,
            })

        self.data = pd.DataFrame(data_list)
        print(f"Loaded {len(self.data)} samples from Point-Bench")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'image_path': item['image_path'],
            'mask_path': item['mask_path'],
            'question': item['question'],
            'category': item['category'],
            'count': item['count'],
        }

    def build_prompt(self, line):
        """Build prompt for pointing task."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']

        # Build message with image and query
        msgs = []
        if os.path.exists(image_path):
            msgs.append(dict(type='image', value=image_path))

        # Add pointing instruction
        prompt_text = f"{question}\nPoint to the relevant location(s)."
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def dump_image(self, line):
        """Return image path."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['image_path']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        Point-Bench evaluation: check if predicted points fall within mask.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

        # Track per-category accuracy
        cat_correct = {}
        cat_total = {}

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            mask_path = row.get('mask_path', '')
            image_path = row.get('image_path', '')
            category = row.get('category', 'unknown')
            expected_count = row.get('count', 1)

            is_correct = False

            # Extract predicted points
            points = extract_molmo_points(pred_text)
            if not points:
                points = extract_points_robust(pred_text)

            # Load mask and image for dimensions
            if points and os.path.exists(mask_path) and os.path.exists(image_path):
                try:
                    mask = Image.open(mask_path)
                    image = Image.open(image_path)
                    width, height = image.size

                    # Check if any point is in mask
                    for (x, y) in points:
                        if check_point_in_mask(x, y, mask, width, height):
                            is_correct = True
                            break
                except Exception as e:
                    pass

            if is_correct:
                correct += 1
            total += 1

            # Track per-category
            if category not in cat_correct:
                cat_correct[category] = 0
                cat_total[category] = 0
            if is_correct:
                cat_correct[category] += 1
            cat_total[category] += 1

            results.append({
                'index': row.get('index', idx),
                'category': category,
                'prediction': pred_text,
                'num_points': len(points),
                'correct': is_correct,
            })

        # Calculate accuracy
        accuracy = correct / total * 100 if total > 0 else 0

        # Calculate per-category accuracy
        cat_accuracy = {}
        for cat in cat_correct:
            if cat_total[cat] > 0:
                cat_accuracy[cat] = cat_correct[cat] / cat_total[cat] * 100

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
        # Add per-category accuracy
        for cat, acc in cat_accuracy.items():
            summary[f'{cat}_accuracy'] = acc

        return summary
