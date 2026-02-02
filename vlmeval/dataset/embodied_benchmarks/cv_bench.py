"""
CV-Bench Dataset Implementation

CV-Bench is a vision-language benchmark from NYU.
- Dataset: HuggingFace `nyu-visionx/CV-Bench`
- Split: "test"
- Format: Pre-formatted prompts with MCQ options
- Evaluation: Exact match of answer letter like "(C)"
"""

import os
import pandas as pd
from PIL import Image
from datasets import load_dataset
from ..image_base import ImageBaseDataset
from ...smp import load, dump, LMUDataRoot
from .utils import extract_answer_letter


class CVBenchDataset(ImageBaseDataset):
    """CV-Bench: Computer Vision Benchmark for VLMs."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    DATASET_URL = {}  # Loaded from HuggingFace
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['CVBench_Embodied']

    def __init__(self, dataset='CVBench_Embodied', **kwargs):
        self.dataset_name = dataset
        # Setup image root directory
        ROOT = LMUDataRoot()
        self.img_root = os.path.join(ROOT, 'images', 'CVBench_Embodied')
        os.makedirs(self.img_root, exist_ok=True)
        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        dataset_path = "nyu-visionx/CV-Bench"

        try:
            hf_dataset = load_dataset(dataset_path, split="test")
        except Exception:
            hf_dataset = load_dataset(dataset_path, split="train")

        # Convert to DataFrame format expected by VLMEvalKit
        data_list = []
        for idx, item in enumerate(hf_dataset):
            # Save image to disk
            img_path = os.path.join(self.img_root, f"{idx}.jpg")
            if not os.path.exists(img_path):
                pil_image = item['image']
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_image.save(img_path, 'JPEG')

            data_list.append({
                'index': idx,
                'image_path': img_path,
                'question': item['prompt'],  # Pre-formatted prompt
                'answer': item['answer'],  # Format: "(C)"
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
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build prompt for model input.

        CV-Bench uses pre-formatted prompts from the dataset.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Get image path
        image_path = line['image_path']

        # Use pre-formatted prompt directly
        prompt = line['question']

        # Return message format
        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=prompt),
        ]
        return msgs

    def dump_image(self, line):
        """Return image path for the sample."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [line['image_path']]

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        CV-Bench evaluation: exact match of answer letter like "(C)".
        """
        data = load(eval_file)

        # Ensure required columns exist
        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))

            # Extract answer letter from prediction
            pred_letter = extract_answer_letter(pred_text, max_letter='H')

            # Check correctness
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

        # Calculate accuracy
        accuracy = correct / total * 100 if total > 0 else 0

        # Save detailed results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        # Return summary
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
