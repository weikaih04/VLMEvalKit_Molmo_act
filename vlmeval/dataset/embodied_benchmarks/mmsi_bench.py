"""
MMSI-Bench Dataset Implementation

MMSI-Bench is a Multi-Image Spatial Intelligence benchmark.
- Dataset: HuggingFace `RunsenXu/MMSI-Bench`
- Format: Multi-image MCQ (4 options A-D)
- Tasks: 10 spatial reasoning types + multi-step reasoning
- Evaluation: Exact match of answer letter

Reference: https://github.com/OpenRobotLab/MMSI-Bench
"""

import os
import io
import pandas as pd
from PIL import Image
from datasets import load_dataset
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter
from . import EMBODIED_DATA_ROOT


def extract_mmsi_answer(text):
    """Extract answer using MMSI-Bench's official extraction logic.

    Priority:
    1. Double backticks: ``A``
    2. Single backticks: `A`
    3. Word boundary regex: isolated A-D letters
    """
    import re

    if not text:
        return None

    # Strategy 1: Double backticks ``A``
    match = re.search(r'``([A-D])``', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 2: Single backticks `A`
    match = re.search(r'`([A-D])`', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 3: Word boundary - isolated letter
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1).upper()

    # Fallback to general extraction
    return extract_answer_letter(text, max_letter='D')


class MMSIBenchDataset(ImageBaseDataset):
    """MMSI-Bench: Multi-Image Spatial Intelligence Benchmark."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['MMSI_Bench_Embodied', 'MMSI_Bench']

    def __init__(self, dataset='MMSI_Bench_Embodied', **kwargs):
        self.dataset_name = dataset

        # Local cache for decoded images
        self.image_cache_dir = kwargs.get(
            'image_cache_dir',
            os.path.join(EMBODIED_DATA_ROOT, 'mmsi_bench', 'images')
        )
        os.makedirs(self.image_cache_dir, exist_ok=True)

        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        try:
            hf_dataset = load_dataset("RunsenXu/MMSI-Bench", split="test")
        except Exception:
            try:
                hf_dataset = load_dataset("RunsenXu/MMSI-Bench", split="train")
            except Exception as e:
                print(f"Warning: Could not load MMSI-Bench: {e}")
                self.data = pd.DataFrame()
                return

        data_list = []
        for idx, item in enumerate(hf_dataset):
            question = item['question']
            answer = item['answer']  # A, B, C, or D
            question_type = item.get('question_type', 'general')
            difficulty = item.get('difficulty', 'unknown')

            # Decode and save images
            image_paths = []
            images_data = item.get('images', [])

            for img_idx, img_bytes in enumerate(images_data):
                img_path = os.path.join(self.image_cache_dir, f"{idx}_{img_idx}.jpg")

                if not os.path.exists(img_path):
                    try:
                        # Decode binary image data
                        if isinstance(img_bytes, bytes):
                            img = Image.open(io.BytesIO(img_bytes))
                        elif isinstance(img_bytes, dict) and 'bytes' in img_bytes:
                            img = Image.open(io.BytesIO(img_bytes['bytes']))
                        elif isinstance(img_bytes, Image.Image):
                            img = img_bytes
                        else:
                            continue
                        img.convert('RGB').save(img_path, 'JPEG')
                    except Exception as e:
                        print(f"Warning: Could not decode image {idx}_{img_idx}: {e}")
                        continue

                image_paths.append(img_path)

            if not image_paths:
                continue

            data_list.append({
                'index': idx,
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'difficulty': difficulty,
                'image_paths': image_paths,
            })

        self.data = pd.DataFrame(data_list)
        print(f"Loaded {len(self.data)} samples from MMSI-Bench")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'question': item['question'],
            'answer': item['answer'],
            'question_type': item['question_type'],
            'difficulty': item['difficulty'],
            'image_paths': item['image_paths'],
        }

    def build_prompt(self, line):
        """Build prompt with multiple images."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = line['image_paths']
        question = line['question']

        # Build message: [IMG1, IMG2, ..., TEXT]
        msgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                msgs.append(dict(type='image', value=img_path))

        # Add question with instruction (matching MMSI-Bench official format)
        prompt_text = f"{question}\nAnswer with the option's letter from the given choices directly. Enclose the option's letter within ``."
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def dump_image(self, line):
        """Return image paths."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['image_paths']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        MMSI-Bench evaluation: exact match of answer letter (A-D).
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
            gt_answer = str(row.get('answer', '')).upper()
            q_type = row.get('question_type', 'general')

            # Extract answer letter using MMSI-Bench's official logic
            pred_letter = extract_mmsi_answer(pred_text)

            is_correct = pred_letter is not None and pred_letter.upper() == gt_answer

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
                'parsed_prediction': pred_letter,
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
