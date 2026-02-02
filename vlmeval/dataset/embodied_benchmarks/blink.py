"""
BLINK Dataset Implementation

BLINK is a visual perception benchmark with 14 subtasks.
- Dataset: HuggingFace `BLINK-Benchmark/BLINK`
- Format: MCQ with multiple images (up to 4)
- Evaluation: Exact match, average over subtasks
"""

import pandas as pd
from datasets import load_dataset
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import extract_answer_letter


# All 14 BLINK subtasks
BLINK_SUBTASKS = [
    'Art_Style',
    'Functional_Correspondence',
    'Multi-view_Reasoning',
    'Relative_Reflectance',
    'Visual_Correspondence',
    'Counting',
    'IQ_Test',
    'Object_Localization',
    'Semantic_Correspondence',
    'Visual_Similarity',
    'Forensic_Detection',
    'Jigsaw',
    'Relative_Depth',
    'Spatial_Relation',
]


class BLINKDataset(ImageBaseDataset):
    """BLINK: Visual Perception Benchmark with 14 Subtasks."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        # Support individual subtasks and combined
        datasets = ['BLINK_Embodied']
        for task in BLINK_SUBTASKS:
            datasets.append(f'BLINK_{task}')
        return datasets

    def __init__(self, dataset='BLINK_Embodied', split='val', **kwargs):
        self.dataset_name = dataset
        self.split = split

        # Determine which subtasks to load
        if dataset == 'BLINK_Embodied':
            self.subtasks = BLINK_SUBTASKS
        else:
            # Extract subtask name from dataset name
            task_name = dataset.replace('BLINK_', '')
            if task_name in BLINK_SUBTASKS:
                self.subtasks = [task_name]
            else:
                self.subtasks = BLINK_SUBTASKS

        self._load_hf_dataset()

    def _load_hf_dataset(self):
        """Load dataset from HuggingFace."""
        dataset_path = "BLINK-Benchmark/BLINK"

        data_list = []
        idx_counter = 0

        for task_name in self.subtasks:
            try:
                hf_dataset = load_dataset(dataset_path, task_name, split=self.split)
            except Exception:
                try:
                    hf_dataset = load_dataset(dataset_path, task_name, split='test')
                except Exception:
                    print(f"Warning: Could not load BLINK subtask {task_name}")
                    continue

            for item in hf_dataset:
                # Collect images (up to 4)
                images = []
                for k in ['image_1', 'image_2', 'image_3', 'image_4']:
                    if k in item and item[k] is not None:
                        images.append(item[k])

                if not images:
                    continue

                # Get prompt and answer
                prompt = item.get('prompt', '')
                answer = item.get('answer', '')  # Format: "(A)"
                choices = item.get('choices', [])

                data_list.append({
                    'index': idx_counter,
                    'idx': item.get('idx', str(idx_counter)),
                    'subtask': task_name,
                    'images': images,
                    'question': prompt,
                    'choices': choices,
                    'answer': answer,
                })
                idx_counter += 1

        self.data = pd.DataFrame(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'idx': item['idx'],
            'subtask': item['subtask'],
            'images': item['images'],
            'question': item['question'],
            'answer': item['answer'],
        }

    def build_prompt(self, line):
        """Build prompt with multiple images."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        images = line['images']
        prompt = line['question']  # Pre-formatted with options

        # Build message: [IMG1, IMG2, ..., TEXT]
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
        """Evaluate predictions.

        BLINK evaluation: accuracy per subtask, then average.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        # Track per-subtask accuracy
        subtask_correct = {task: 0 for task in BLINK_SUBTASKS}
        subtask_total = {task: 0 for task in BLINK_SUBTASKS}

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))
            subtask = row.get('subtask', 'unknown')

            # Extract answer letter
            pred_letter = extract_answer_letter(pred_text, max_letter='E')

            is_correct = pred_letter is not None and pred_letter == gt_answer

            if subtask in subtask_correct:
                if is_correct:
                    subtask_correct[subtask] += 1
                subtask_total[subtask] += 1

            results.append({
                'index': row.get('index', idx),
                'subtask': subtask,
                'prediction': pred_text,
                'parsed_prediction': pred_letter,
                'answer': gt_answer,
                'correct': is_correct,
            })

        # Calculate per-subtask accuracy
        subtask_accuracy = {}
        for task in BLINK_SUBTASKS:
            if subtask_total[task] > 0:
                subtask_accuracy[task] = subtask_correct[task] / subtask_total[task] * 100
            else:
                subtask_accuracy[task] = 0.0

        # Calculate average accuracy (average over subtasks)
        valid_tasks = [task for task in BLINK_SUBTASKS if subtask_total[task] > 0]
        if valid_tasks:
            average_accuracy = sum(subtask_accuracy[task] for task in valid_tasks) / len(valid_tasks)
        else:
            average_accuracy = 0.0

        # Save detailed results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        # Return summary with per-subtask breakdown
        summary = {
            'average_accuracy': average_accuracy,
            **{f'{task}_accuracy': subtask_accuracy[task] for task in valid_tasks},
        }

        return summary
