"""
ERQA Dataset Implementation

ERQA is an image VQA benchmark with interleaved images and text.
- Dataset: TFRecord format
- Format: VQA with multiple images interleaved in question text
- Evaluation: Normalized exact match

Requires TensorFlow for reading TFRecord files.
"""

import os
import io
import numpy as np
import pandas as pd
from PIL import Image
from ..image_base import ImageBaseDataset
from ...smp import load, dump
from .utils import normalize_text
from . import EMBODIED_DATA_ROOT


def hide_tf_gpu():
    """Hide GPU from TensorFlow to prevent memory conflicts."""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except ImportError:
        pass


def parse_tfrecord(tfrecord_path):
    """Parse ERQA TFRecord file."""
    hide_tf_gpu()

    try:
        import tensorflow as tf
    except ImportError:
        print("Warning: TensorFlow is required to read ERQA TFRecord files")
        return []

    feature_description = {
        'answer': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'question_type': tf.io.VarLenFeature(tf.string),
        'visual_indices': tf.io.VarLenFeature(tf.int64),
        'question': tf.io.FixedLenFeature([], tf.string)
    }

    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        parsed['visual_indices'] = tf.sparse.to_dense(parsed['visual_indices'])
        parsed['image/encoded'] = tf.sparse.to_dense(parsed['image/encoded'])
        parsed['question_type'] = tf.sparse.to_dense(parsed['question_type'])
        return parsed

    if not os.path.exists(tfrecord_path):
        return []

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)

    samples = []
    for example in dataset:
        question = example['question'].numpy().decode('utf-8')
        answer = example['answer'].numpy().decode('utf-8')
        visual_indices = example['visual_indices'].numpy().tolist()

        # Decode images
        images = []
        for img_bytes in example['image/encoded']:
            try:
                img_data = img_bytes.numpy()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(img)
            except Exception:
                pass

        samples.append({
            'question': question,
            'answer': answer,
            'images': images,
            'visual_indices': visual_indices,
        })

    return samples


def construct_interleaved_content(question, images, visual_indices):
    """Construct interleaved image+text content based on visual_indices.

    visual_indices specifies where each image should be inserted in the text.
    """
    content = []

    # Pair images with their indices
    img_pairs = list(zip(images, visual_indices)) if visual_indices else []
    img_pairs.sort(key=lambda x: x[1])

    # Case A: No indices or empty
    if not img_pairs:
        for img in images:
            content.append({'type': 'image', 'image': img})
        content.append({'type': 'text', 'text': question})
        return content

    # Case B: All indices are 0 (images first)
    if all(idx == 0 for _, idx in img_pairs):
        for img, _ in img_pairs:
            content.append({'type': 'image', 'image': img})
        content.append({'type': 'text', 'text': question})
        return content

    # Case C: Interleaved
    last_pos = 0
    for img, idx in img_pairs:
        # Add text before this image
        if idx > last_pos:
            text_segment = question[last_pos:idx]
            if text_segment:
                content.append({'type': 'text', 'text': text_segment})

        # Add the image
        content.append({'type': 'image', 'image': img})
        last_pos = idx

    # Add remaining text
    if last_pos < len(question):
        content.append({'type': 'text', 'text': question[last_pos:]})

    return content


class ERQADataset(ImageBaseDataset):
    """ERQA: Embodied Robotic Question Answering.

    Multi-image VQA with interleaved images and text.
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['ERQA_Embodied', 'ERQA']

    def __init__(self, dataset='ERQA_Embodied', **kwargs):
        self.dataset_name = dataset

        # TFRecord path
        default_path = os.path.join(EMBODIED_DATA_ROOT, 'erqa', 'erqa.tfrecord')
        self.tfrecord_path = kwargs.get('tfrecord_path', default_path)

        self._load_tfrecord_dataset()

    def _load_tfrecord_dataset(self):
        """Load dataset from TFRecord file."""
        samples = parse_tfrecord(self.tfrecord_path)

        if not samples:
            print(f"Warning: No samples loaded from {self.tfrecord_path}")
            self.data = pd.DataFrame()
            return

        data_list = []
        for idx, sample in enumerate(samples):
            img_count = len(sample['images'])
            category = "Single-Image" if img_count == 1 else "Multi-Image"

            data_list.append({
                'index': idx,
                'images': sample['images'],
                'visual_indices': sample['visual_indices'],
                'question': sample['question'],
                'answer': sample['answer'],
                'category': category,
            })

        self.data = pd.DataFrame(data_list)
        print(f"ERQA: Loaded {len(self.data)} samples from TFRecord")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'images': item['images'],
            'visual_indices': item['visual_indices'],
            'question': item['question'],
            'answer': item['answer'],
            'category': item['category'],
        }

    def build_prompt(self, line):
        """Build interleaved image+text prompt."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        images = line['images']
        visual_indices = line['visual_indices']
        question = line['question']

        # Construct interleaved content
        content = construct_interleaved_content(question, images, visual_indices)

        # Convert to VLMEvalKit message format
        msgs = []
        for item in content:
            if item['type'] == 'image':
                msgs.append(dict(type='image', value=item['image']))
            else:
                msgs.append(dict(type='text', value=item['text']))

        return msgs

    def dump_image(self, line):
        """Return images."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        return line['images']

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions.

        ERQA evaluation: normalized exact match.
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        correct = 0
        total = 0

        # Track per-category accuracy
        cat_correct = {"Single-Image": 0, "Multi-Image": 0}
        cat_total = {"Single-Image": 0, "Multi-Image": 0}

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))
            category = row.get('category', 'unknown')

            # Normalized exact match
            pred_clean = pred_text.replace(".", "").strip().lower()
            gt_clean = gt_answer.replace(".", "").strip().lower()

            is_correct = (pred_clean == gt_clean)

            if is_correct:
                correct += 1
            total += 1

            # Track per-category
            if category in cat_correct:
                if is_correct:
                    cat_correct[category] += 1
                cat_total[category] += 1

            results.append({
                'index': row.get('index', idx),
                'category': category,
                'prediction': pred_text,
                'pred_normalized': pred_clean,
                'answer': gt_answer,
                'answer_normalized': gt_clean,
                'correct': is_correct,
            })

        # Calculate overall accuracy
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
        for cat, acc in cat_accuracy.items():
            summary[f'{cat.replace("-", "_")}_accuracy'] = acc

        return summary
