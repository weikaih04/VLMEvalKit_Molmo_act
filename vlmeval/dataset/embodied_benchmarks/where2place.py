"""
Where2Place Dataset Implementation

Where2Place is a spatial placement benchmark.
- Dataset: Local JSONL files (point_questions.jsonl, bbox_questions.jsonl)
- Format: Predict point/bbox, evaluate pixel precision
- Evaluation: Percentage of predicted pixels inside mask
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import text2pts_official, calculate_accuracy_official, get_model_type, extract_points_robust
from . import EMBODIED_DATA_ROOT


class Where2PlaceDataset(ImageBaseDataset):
    """Where2Place: Spatial Placement Benchmark."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['Where2Place_point', 'Where2Place_bbox', 'Where2Place_Embodied']

    def __init__(self, dataset='Where2Place_Embodied', **kwargs):
        self.dataset_name = dataset

        # Determine task type
        if 'point' in dataset.lower():
            self.task = 'point'
        elif 'bbox' in dataset.lower():
            self.task = 'bbox'
        else:
            self.task = 'point'  # default

        self._load_local_dataset()

    def _load_local_dataset(self):
        """Load dataset from local JSONL file."""
        data_dir = os.path.join(EMBODIED_DATA_ROOT, 'where2place')

        if self.task == 'point':
            json_filename = 'point_questions.jsonl'
        else:
            json_filename = 'bbox_questions.jsonl'

        json_path = os.path.join(data_dir, json_filename)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Where2Place JSONL not found at: {json_path}")

        with open(json_path, 'r') as f:
            questions = [json.loads(line) for line in f]

        data_list = []
        for idx, q_item in enumerate(questions):
            image_file = q_item['image']
            query = q_item['text']

            # Load image and mask
            img_path = os.path.join(data_dir, 'images', image_file)

            # Mask handling: try filename first, then index-based
            mask_path_name = os.path.join(data_dir, 'masks', image_file)
            mask_path_idx = os.path.join(data_dir, 'masks', f'{idx:02d}.jpg')

            if os.path.exists(mask_path_name):
                mask_path = mask_path_name
            elif os.path.exists(mask_path_idx):
                mask_path = mask_path_idx
            else:
                continue  # Skip if mask not found

            if not os.path.exists(img_path):
                continue  # Skip if image not found

            try:
                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                img_width = image.width
                img_height = image.height
            except Exception:
                continue

            data_list.append({
                'index': idx,
                'mask': mask,
                'question': query,
                'image_path': img_path,
                'mask_path': mask_path,
                'image_width': img_width,
                'image_height': img_height,
            })

        self.data = pd.DataFrame(data_list)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'image_path': item['image_path'],
            'mask': item['mask'],
            'question': item['question'],
        }

    def build_prompt(self, line, model_name=None):
        """Build placement prompt with model-specific format."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']

        model_type = get_model_type(model_name)

        if model_type == 'molmo':
            prompt_text = question
        elif model_type == 'qwen3':
            prompt_text = question + '\nLocate the placement position with a point and report its point coordinates in JSON format like this: {"point_2d": [x, y], "label": "object/region"}'
        elif model_type == 'qwen25':
            prompt_text = question + "\nOutput the coordinates in XML format <points x y>object</points>."
        elif model_type == 'internvl':
            prompt_text = question + "\nRespond with ONLY the coordinates in [x, y] format."
        elif model_type == 'llava':
            # PointArena official prompt for LLaVA-OneVision
            try:
                from PIL import Image as PILImage
                img = PILImage.open(image_path)
                img_width, img_height = img.size
            except Exception:
                img_width, img_height = 0, 0
            prompt_text = (
                f"{question} The image dimensions are width={img_width}px, height={img_height}px.\n"
                "Give EXACT PIXEL COORDINATES in [x, y] format, where x is horizontal (left-to-right) "
                "and y is vertical (top-to-bottom). ONLY return the coordinates with no additional text or explanations."
            )
        else:
            # phi4, etc.
            prompt_text = (
                question + "\nGive EXACT PIXEL COORDINATES in [x, y] format, "
                "where x is horizontal and y is vertical. "
                "ONLY return coordinates with no additional text."
            )

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=prompt_text),
        ]
        return msgs

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['image_path']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate placement predictions.

        Where2Place evaluation: pixel precision (% of predicted pixels in mask).
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        scores = []
        results = []

        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))

            # Get mask and image dimensions from original data
            orig_idx = row.get('index', idx)
            if orig_idx < len(self.data):
                orig_row = self.data.iloc[orig_idx]
                mask = orig_row['mask']
                img_width = orig_row['image_width']
                img_height = orig_row['image_height']
            else:
                results.append({
                    'index': orig_idx,
                    'prediction': pred_text,
                    'accuracy': 0.0,
                })
                scores.append(0.0)
                continue

            # Parse points from prediction using robust extractor, fallback to text2pts for bbox
            robust_pts = extract_points_robust(pred_text, img_width, img_height)
            if robust_pts:
                # Convert from 0-100 percentage to pixel coordinates
                points = np.array([[int(x / 100.0 * img_width), int(y / 100.0 * img_height)] for x, y in robust_pts])
            else:
                points = text2pts_official(pred_text, img_width, img_height)

            # Calculate pixel precision
            acc = calculate_accuracy_official(points, mask)
            scores.append(acc)

            results.append({
                'index': orig_idx,
                'prediction': pred_text,
                'num_points': len(points) if len(points) > 0 else 0,
                'accuracy': acc,
            })

        # Calculate average accuracy
        average_accuracy = np.mean(scores) * 100 if scores else 0

        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        return {
            'accuracy': average_accuracy,
            'total': len(scores),
        }
