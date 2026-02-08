"""
RoboSpatial-Home Dataset Implementation

RoboSpatial-Home has two evaluation modes:
1. Pointing: Check if predicted point is in mask
2. VQA: Substring matching for answers

- Dataset: HuggingFace `chanhee-luke/RoboSpatial-Home`
- Pointing splits: ["context"]
- VQA splits: ["configuration", "compatibility"]
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_molmo_points, extract_points_robust, check_point_in_mask, normalize_text, format_pointing_prompt, get_model_type
from . import EMBODIED_DATA_ROOT


def convert_pointing_prompt(original_question: str, model_name: str = None) -> str:
    """
    Remove the output format requirement and add model-specific pointing instruction.

    Original format:
    "In the image, there is a bowl. Pinpoint several points within the
    vacant space situated to the left of the bowl. Your answer should
    be formatted as a list of tuples, i.e. [(x1, y1), ...]..."
    """
    # Remove the format requirement part
    cleaned = re.sub(
        r'\s*Your answer should be formatted as a list of tuples.*$',
        '',
        original_question,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    model_type = get_model_type(model_name)

    if model_type == 'molmo':
        return cleaned + " Point to the areas that the prompt asked."
    elif model_type == 'qwen3':
        return cleaned + '\nLocate the relevant areas with points and report their point coordinates in JSON format like this: {"point_2d": [x, y], "label": "object/region"}'
    elif model_type == 'qwen25':
        return cleaned + "\nOutput the coordinates in XML format <points x y>object</points>."
    elif model_type == 'internvl':
        return cleaned + " Point to the areas that the prompt asked."
    else:
        # llava, internvl, phi4
        return (
            cleaned + "\nGive EXACT PIXEL COORDINATES in [x, y] format, "
            "where x is horizontal and y is vertical. "
            "ONLY return coordinates with no additional text."
        )


class RoboSpatialPointingDataset(ImageBaseDataset):
    """RoboSpatial-Home: Pointing Task."""

    TYPE = 'VQA'  # Pointing task
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['RoboSpatial_Pointing']

    def __init__(self, dataset='RoboSpatial_Pointing', **kwargs):
        self.dataset_name = dataset
        self.splits = ['context']
        # Directory to save images
        self.img_root = os.path.join(EMBODIED_DATA_ROOT, 'robospatial', 'pointing_images')
        os.makedirs(self.img_root, exist_ok=True)
        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        dataset_path = "chanhee-luke/RoboSpatial-Home"

        datasets = []
        for split in self.splits:
            try:
                ds = load_dataset(dataset_path, split=split)
                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Could not load RoboSpatial split {split}: {e}")

        if not datasets:
            self.data = pd.DataFrame()
            return

        if len(datasets) > 1:
            hf_dataset = concatenate_datasets(datasets)
        else:
            hf_dataset = datasets[0]

        data_list = []
        for idx, item in enumerate(hf_dataset):
            # Save image to disk
            img_path = os.path.join(self.img_root, f'{idx:06d}.jpg')
            if not os.path.exists(img_path):
                try:
                    item['img'].convert('RGB').save(img_path, 'JPEG')
                except Exception:
                    continue

            data_list.append({
                'index': idx,
                'image_path': img_path,
                'mask': item.get('mask'),
                'question': item['question'],
                'image_width': item['img'].width,
                'image_height': item['img'].height,
            })

        self.data = pd.DataFrame(data_list)

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
        """Build pointing prompt with model-specific format."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']

        model_type = get_model_type(model_name)
        if model_type == 'llava':
            # PointArena official prompt for LLaVA-OneVision
            cleaned = re.sub(
                r'\s*Your answer should be formatted as a list of tuples.*$',
                '', question, flags=re.IGNORECASE | re.DOTALL
            ).strip()
            try:
                from PIL import Image as PILImage
                img = PILImage.open(image_path)
                img_width, img_height = img.size
            except Exception:
                img_width, img_height = 0, 0
            prompt = (
                f"{cleaned} The image dimensions are width={img_width}px, height={img_height}px.\n"
                "Give EXACT PIXEL COORDINATES in [x, y] format, where x is horizontal (left-to-right) "
                "and y is vertical (top-to-bottom). ONLY return the coordinates with no additional text or explanations."
            )
        else:
            prompt = convert_pointing_prompt(question, model_name)

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=prompt),
        ]
        return msgs

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['image_path']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate pointing predictions.

        RoboSpatial-Pointing: AT LEAST ONE predicted point must be in mask.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

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
                    'correct': False,
                })
                continue

            # Extract points - try Molmo format first, then robust fallback
            points = extract_molmo_points(pred_text)
            if not points:
                points = extract_points_robust(pred_text)

            # Check if AT LEAST ONE point is in mask
            is_hit = False
            if points and mask is not None:
                for pt in points:
                    if check_point_in_mask(pt[0], pt[1], mask, img_width, img_height):
                        is_hit = True
                        break

            if is_hit:
                correct += 1
            total += 1

            results.append({
                'index': orig_idx,
                'prediction': pred_text,
                'parsed_points': points,
                'correct': is_hit,
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


class RoboSpatialVQADataset(ImageBaseDataset):
    """RoboSpatial-Home: VQA Task."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['RoboSpatial_VQA']

    def __init__(self, dataset='RoboSpatial_VQA', **kwargs):
        self.dataset_name = dataset
        self.splits = ['configuration', 'compatibility']
        # Directory to save images
        self.img_root = os.path.join(EMBODIED_DATA_ROOT, 'robospatial', 'vqa_images')
        os.makedirs(self.img_root, exist_ok=True)
        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        dataset_path = "chanhee-luke/RoboSpatial-Home"

        datasets = []
        for split in self.splits:
            try:
                ds = load_dataset(dataset_path, split=split)
                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Could not load RoboSpatial split {split}: {e}")

        if not datasets:
            self.data = pd.DataFrame()
            return

        if len(datasets) > 1:
            hf_dataset = concatenate_datasets(datasets)
        else:
            hf_dataset = datasets[0]

        data_list = []
        for idx, item in enumerate(hf_dataset):
            # Save image to disk
            img_path = os.path.join(self.img_root, f'{idx:06d}.jpg')
            if not os.path.exists(img_path):
                try:
                    item['img'].convert('RGB').save(img_path, 'JPEG')
                except Exception:
                    continue

            data_list.append({
                'index': idx,
                'image_path': img_path,
                'question': item['question'],
                'answer': item['answer'],
            })

        self.data = pd.DataFrame(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'image_path': item['image_path'],
            'question': item['question'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build VQA prompt."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=question),
        ]
        return msgs

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['image_path']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate VQA predictions.

        RoboSpatial-VQA: Normalized GT substring in normalized prediction.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))

            # Normalize and check substring
            pred_norm = normalize_text(pred_text)
            gt_norm = normalize_text(gt_answer)

            is_correct = gt_norm in pred_norm if gt_norm else False

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'index': row.get('index', idx),
                'prediction': pred_text,
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
