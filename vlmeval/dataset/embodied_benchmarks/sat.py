"""
SAT Dataset Implementation

SAT is a spatial understanding benchmark with circular evaluation.
- Dataset: HuggingFace `FlagEval/SAT` or `array/SAT`
- Split: "test"
- Format: MCQ with multiple images, options are SHUFFLED
- Evaluation: Exact match after shuffling (circular evaluation)
"""

import io
import random
import pandas as pd
from PIL import Image
from datasets import load_dataset
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter


class SATDataset(ImageBaseDataset):
    """SAT: Spatial Action Understanding Test with Circular Evaluation."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['SAT_Embodied']

    def __init__(self, dataset='SAT_Embodied', seed=42, **kwargs):
        self.dataset_name = dataset
        self.seed = seed
        random.seed(seed)
        self._load_hf_dataset()

    def _decode_images(self, item):
        """Handle various image storage formats in SAT."""
        images = []

        # Case A: 'images' column
        if 'images' in item and item['images']:
            val = item['images']
            if isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], bytes):
                    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in val]
                else:
                    images = [img.convert("RGB") if hasattr(img, 'convert') else img for img in val]

        # Case B: 'image_bytes' column
        elif 'image_bytes' in item and item['image_bytes']:
            images = [Image.open(io.BytesIO(b)).convert("RGB") for b in item['image_bytes']]

        # Case C: Single image fallback
        elif 'image' in item and item['image']:
            img = item['image']
            if isinstance(img, bytes):
                images = [Image.open(io.BytesIO(img)).convert("RGB")]
            else:
                images = [img.convert("RGB") if hasattr(img, 'convert') else img]

        return images

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        try:
            hf_dataset = load_dataset("FlagEval/SAT", split="test")
        except Exception:
            try:
                hf_dataset = load_dataset("array/SAT", split="test")
            except Exception:
                hf_dataset = load_dataset("FlagEval/SAT", split="train")

        data_list = []
        for idx, item in enumerate(hf_dataset):
            images = self._decode_images(item)
            if not images:
                continue

            question = item['question']
            raw_options = list(item['answers'])
            gt_text = item['correct_answer']

            # Shuffle options for circular evaluation
            shuffled_options = raw_options.copy()
            random.shuffle(shuffled_options)

            # Find GT index in shuffled options
            try:
                gt_index = shuffled_options.index(gt_text)
                gt_letter = f"({chr(ord('A') + gt_index)})"
            except ValueError:
                continue

            data_list.append({
                'index': idx,
                'images': images,
                'question': question,
                'options': shuffled_options,
                'original_options': raw_options,
                'correct_answer_text': gt_text,
                'answer': gt_letter,
            })

        self.data = pd.DataFrame(data_list)
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'images': item['images'],
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build MCQ prompt with shuffled options."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        images = line['images']
        question = line['question']
        options = line['options']

        # Build prompt
        letters = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        formatted_options = []
        for i, opt in enumerate(options):
            if i < len(letters):
                formatted_options.append(f"{letters[i]} {opt}")

        prompt = f"{question}\n" + "\n".join(formatted_options) + "\nAnswer with the option letter."

        # Build message with multiple images
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

            pred_letter = extract_answer_letter(pred_text, max_letter='E')

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
