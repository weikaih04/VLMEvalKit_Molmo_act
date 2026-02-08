"""
OpenEQA Dataset Implementation

OpenEQA is an open-ended video QA benchmark.
- Dataset: JSON metadata + local frame folders or video files
- Format: Video VQA (open-ended)
- Evaluation: Only saves predictions (no automatic scoring)

Data structure:
  {data_root}/frames/{episode_history}/00001.jpg, 00002.jpg, ...
  {data_root}/videos/{episode_history}.mp4  (optional, for video mode)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
from ..video_base import VideoBaseDataset
from ...smp import load, dump
from . import EMBODIED_DATA_ROOT


def get_frames_from_folder(folder_path, num_frames=8):
    """Load frames from a folder of images."""
    if not folder_path or not os.path.exists(folder_path):
        return []

    # Find all images
    files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )

    if not files:
        return []

    # Uniformly sample indices
    indices = np.linspace(0, len(files) - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        try:
            img = Image.open(files[idx]).convert("RGB")
            frames.append(img)
        except Exception:
            pass

    return frames


def extract_frames_from_video(video_path, num_frames=8, output_dir=None):
    """Extract frames from a video file and save as JPEG files.

    Returns list of file paths to extracted frames.
    """
    from decord import VideoReader, cpu

    if not video_path or not os.path.exists(video_path):
        return []

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), '_frames_cache',
                                  os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_dir, exist_ok=True)

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames, dtype=int)

        frame_paths = []
        for i, idx in enumerate(indices):
            pth = os.path.join(output_dir, f'{i:05d}.jpg')
            if not os.path.exists(pth):
                frame = Image.fromarray(vr[idx].asnumpy())
                frame.save(pth)
            frame_paths.append(pth)
        return frame_paths
    except Exception as e:
        print(f"Warning: Failed to extract frames from {video_path}: {e}")
        return []


class OpenEQADataset(VideoBaseDataset):
    """OpenEQA: Open-ended Embodied Question Answering.

    Note: This benchmark only saves predictions. No automatic evaluation is performed.
    """

    TYPE = 'VQA'
    MODALITY = 'VIDEO'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['OpenEQA_Embodied', 'OpenEQA']

    def __init__(self, dataset='OpenEQA_Embodied', **kwargs):
        self.dataset_name = dataset
        self.num_frames = kwargs.get('num_frames', 8)

        # Data root containing frames folder
        self.data_root = kwargs.get('data_root', os.path.join(EMBODIED_DATA_ROOT, 'openeqa'))

        # JSON metadata path
        self.json_path = kwargs.get('json_path', os.path.join(self.data_root, 'open-eqa-v0.json'))

        self._load_local_dataset()

    def _load_local_dataset(self):
        """Load dataset from local JSON and frame folders/videos."""
        if not os.path.exists(self.json_path):
            print(f"Warning: OpenEQA JSON not found at {self.json_path}")
            self.data = pd.DataFrame()
            return

        with open(self.json_path, 'r') as f:
            json_data = json.load(f)

        data_list = []
        idx_counter = 0

        for item in json_data:
            question = item['question']
            gt_answer = item['answer']
            rel_path = item['episode_history']  # e.g. "hm3d-v0/000-hm3d-BFRyYbPCCPE"

            # Construct paths
            frames_path = os.path.join(self.data_root, "frames", rel_path)
            video_path = os.path.join(self.data_root, "videos", rel_path + ".mp4")

            # Include samples that have either frames OR video
            has_frames = os.path.exists(frames_path)
            has_video = os.path.exists(video_path)

            if not has_frames and not has_video:
                continue

            data_list.append({
                'index': idx_counter,
                'episode_history': rel_path,
                'frames_path': frames_path if has_frames else None,
                'video_path': video_path if has_video else None,
                'question': question,
                'answer': gt_answer,
            })
            idx_counter += 1

        self.data = pd.DataFrame(data_list)

        if len(self.data) > 0:
            num_with_video = self.data['video_path'].notna().sum()
            num_with_frames = self.data['frames_path'].notna().sum()
            print(f"OpenEQA: Loaded {len(self.data)} samples ({num_with_video} with video, {num_with_frames} with frames)")
        else:
            print(f"Warning: No valid data found in {self.data_root}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'index': item['index'],
            'episode_history': item['episode_history'],
            'frames_path': item['frames_path'],
            'video_path': item['video_path'],
            'question': item['question'],
            'answer': item['answer'],
        }

    def build_prompt(self, line, video_llm=False):
        """Build prompt with video or image frames.

        Args:
            line: Data row or index
            video_llm: If True and video file exists, use type='video'.
                      Otherwise, use type='image' for each frame.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        video_path = line.get('video_path')
        frames_path = line['frames_path']

        msgs = []

        # Use video mode if video_llm=True and video file exists
        if video_llm and video_path and os.path.exists(video_path):
            msgs.append(dict(type='video', value=video_path))
        else:
            # Fallback to multi-image mode: try frames folder first, then extract from video
            frames = get_frames_from_folder(frames_path, self.num_frames)
            if frames:
                for frame in frames:
                    msgs.append(dict(type='image', value=frame))
            elif video_path and os.path.exists(video_path):
                # No frames folder - extract frames from video file as JPEG paths
                frame_paths = extract_frames_from_video(video_path, self.num_frames)
                for fp in frame_paths:
                    msgs.append(dict(type='image', value=fp))
            else:
                print(f"Warning: No frames or video found for {line.get('episode_history', 'unknown')}")

        msgs.append(dict(type='text', value=question))

        return msgs

    def dump_image(self, line):
        """Return video frames."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        frames_path = line['frames_path']
        return get_frames_from_folder(frames_path, self.num_frames)

    def evaluate(self, eval_file, **judge_kwargs):
        """Save predictions only - no automatic evaluation.

        OpenEQA is an open-ended QA benchmark that requires human evaluation
        or LLM-as-judge for proper scoring.

        Returns:
            dict with prediction/answer pairs saved to result file
        """
        data = load(eval_file)

        assert 'prediction' in data.columns, "Missing 'prediction' column"

        results = []
        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt_answer = str(row.get('answer', ''))
            episode = row.get('episode_history', '')
            question = row.get('question', '')

            results.append({
                'index': row.get('index', idx),
                'episode_history': episode,
                'question': question,
                'prediction': pred_text,
                'answer': gt_answer,
            })

        # Save results
        results_df = pd.DataFrame(results)
        result_file = eval_file.replace('.xlsx', '_result.xlsx').replace('.tsv', '_result.tsv')
        dump(results_df, result_file)

        # Return info about saved predictions
        return {
            'total': len(results),
            'note': 'OpenEQA predictions saved. Manual evaluation required.',
            'result_file': result_file,
        }
