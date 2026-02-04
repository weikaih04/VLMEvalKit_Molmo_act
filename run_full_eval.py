"""
Full evaluation script for all embodied benchmarks with Molmo2 + vLLM.

Usage:
    python run_full_eval.py --model Molmo2-4B --output results/
    python run_full_eval.py --model Molmo2-4B --benchmarks CVBench_Embodied EmbSpatial_Embodied
    python run_full_eval.py --model Molmo2-4B --batch_size 8 --output results/
"""

# Set vLLM environment variables BEFORE any imports
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Speed up multimodal request preprocessing

import argparse
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from vlmeval.dataset import build_dataset
from vlmeval.vlm import Molmo2
from vlmeval.vlm.llava import LLaVA_OneVision_HF
from vlmeval.vlm.qwen2_vl import Qwen2VLChat
from vlmeval.vlm.qwen3_vl import Qwen3VLChat
from vlmeval.vlm.phi4_multimodal import Phi4Multimodal
from vlmeval.vlm.internvl import InternVLChat
from vlmeval.dataset.embodied_benchmarks.utils.point_extraction import (
    extract_molmo_points, extract_points_robust, check_point_in_mask
)
from vlmeval.dataset.embodied_benchmarks.utils.evaluation_utils import (
    extract_answer_letter, normalize_text
)


# All available benchmarks
ALL_BENCHMARKS = [
    'CVBench_Embodied',
    'EmbSpatial_Embodied',
    'SAT_Embodied',
    'BLINK_Embodied',
    'RefSpatial_Embodied',
    'RoboSpatial_Pointing',
    'RoboSpatial_VQA',
    'MindCube_Tiny_Embodied',
    'Where2Place_Embodied',
    'vsibench_16frame',  # Native VLMEvalKit VSIBench with proper MRA evaluation
    'CosmosReason1_Embodied',
    'ERQA_Embodied',
    'MMSI_Bench_Embodied',
    'PointBench_Embodied',  # Point-Bench multimodal pointing benchmark
]

# VSIBench question type categories (from native VLMEvalKit)
VSIBENCH_MCA_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
VSIBENCH_NA_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


def get_eval_type(benchmark_name: str) -> str:
    """Determine evaluation type for a benchmark."""
    if 'Pointing' in benchmark_name or 'RefSpatial' in benchmark_name or 'Where2Place' in benchmark_name or 'PointBench' in benchmark_name:
        return 'pointing'
    elif 'vsibench' in benchmark_name.lower():
        return 'vsibench'  # Native VSIBench with MRA evaluation
    elif 'RoboSpatial_VQA' in benchmark_name:
        return 'substring'
    else:
        return 'mcq'


def evaluate_mcq(pred: str, answer: str) -> bool:
    """Evaluate multiple choice question."""
    answer = answer.strip()
    pred = pred.strip()

    # Check if answer is a number
    if answer.isdigit():
        pred_match = re.search(r'\d+', pred)
        if pred_match:
            return pred_match.group(0) == answer
        return False

    # Use the robust answer extraction from evaluation_utils
    pred_letter = extract_answer_letter(pred, max_letter='H')
    answer_letter = extract_answer_letter(answer, max_letter='H')

    if not pred_letter or not answer_letter:
        # Fallback to substring matching
        return normalize_text(answer) in normalize_text(pred)

    return pred_letter == answer_letter


def evaluate_pointing(pred: str, mask, width: int, height: int) -> bool:
    """Evaluate pointing task."""
    points = extract_molmo_points(pred)
    if not points:
        points = extract_points_robust(pred)

    if not points or mask is None:
        return False

    mask_array = np.array(mask)
    for (x, y) in points:
        if check_point_in_mask(x, y, mask_array, width, height):
            return True
    return False


def evaluate_counting(pred: str, answer: str) -> bool:
    """Evaluate counting task by counting points."""
    import re
    try:
        answer_num = int(answer.strip())
    except ValueError:
        return False

    coords_match = re.search(r'coords="([^"]+)"', pred)
    if coords_match:
        coords_str = coords_match.group(1)
        if ';' in coords_str:
            num_points = coords_str.count(';') + 1
        else:
            num_points = len(re.findall(r'[0-9]+ [0-9]{3,4} [0-9]{3,4}', coords_str))
        return num_points == answer_num

    pred_match = re.search(r'\d+', pred)
    if pred_match:
        return int(pred_match.group(0)) == answer_num
    return False


def evaluate_substring(pred: str, answer: str) -> bool:
    """Evaluate by substring matching."""
    return normalize_text(answer) in normalize_text(pred)


def evaluate_vsibench_mca(pred: str, ans: str) -> float:
    """Evaluate VSIBench MCQ question - exact letter match."""
    pred = pred.strip()
    ans = ans.strip()
    if pred.lower() == ans.lower() or pred.lower().startswith(ans.lower() + "."):
        return 1.0
    return 0.0


def evaluate_vsibench_mra(pred: str, ans: str) -> float:
    """Evaluate VSIBench numerical answer using Mean Relative Accuracy.

    MRA measures how close the prediction is across multiple thresholds.
    Returns score from 0 to 2.
    """
    try:
        # Extract number from prediction
        pred_clean = pred.strip()
        # Try to extract first number from text
        num_match = re.search(r'[-+]?\d*\.?\d+', pred_clean)
        if num_match:
            pred_num = float(num_match.group())
        else:
            pred_num = float(pred_clean)

        ans_num = float(ans)
        if ans_num == 0:
            return 1.0 if pred_num == 0 else 0.0

        acc = 0
        for i in range(20):
            theta = 0.5 + i * 0.05
            if abs(pred_num - ans_num) / abs(ans_num) < 1 - theta:
                acc += 1
        return acc / 10  # Returns 0-2
    except (ValueError, ZeroDivisionError) as e:
        return 0.0


def evaluate_vsibench(pred: str, item: dict) -> float:
    """Evaluate VSIBench sample using official evaluation logic.

    Returns score (0-1 for MCA, 0-2 for NA questions).
    """
    answer = str(item.get('answer', ''))
    q_type = str(item.get('type', ''))

    if q_type in VSIBENCH_MCA_TYPES:
        return evaluate_vsibench_mca(pred, answer)
    elif q_type in VSIBENCH_NA_TYPES:
        return evaluate_vsibench_mra(pred, answer)
    else:
        # Fallback: try MCA first, then MRA
        if len(answer) == 1 and answer.upper() in 'ABCD':
            return evaluate_vsibench_mca(pred, answer)
        else:
            return evaluate_vsibench_mra(pred, answer)


def evaluate_sample(pred: str, item: dict, eval_type: str):
    """Evaluate a single prediction.

    Returns:
        bool for most types, float (0-2) for vsibench
    """
    if eval_type == 'pointing':
        mask = item.get('mask')
        image = item.get('image')

        # Load mask from path if not provided as image
        if mask is None:
            mask_path = item.get('mask_path', '')
            if mask_path and os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path)
                except Exception:
                    pass

        # Get image dimensions
        if image is not None:
            width, height = image.width, image.height
        else:
            image_path = item.get('image_path', '')
            if image_path and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception:
                    width = item.get('image_width', 640)
                    height = item.get('image_height', 480)
            else:
                width = item.get('image_width', 640)
                height = item.get('image_height', 480)

        return evaluate_pointing(pred, mask, width, height)
    elif eval_type == 'vsibench':
        return evaluate_vsibench(pred, item)  # Returns float 0-2
    elif eval_type == 'counting':
        return evaluate_counting(pred, str(item.get('answer', '')))
    elif eval_type == 'substring':
        return evaluate_substring(pred, str(item.get('answer', '')))
    else:  # mcq
        return evaluate_mcq(pred, str(item.get('answer', '')))


def run_benchmark(model, benchmark_name: str, output_dir: str = None, batch_size: int = 16, model_name: str = None):
    """Run evaluation on a single benchmark with vLLM continuous batching."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {benchmark_name}")
    print(f"{'='*60}")

    try:
        ds = build_dataset(benchmark_name)
        total_samples = len(ds.data)
        print(f"Total samples: {total_samples}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    eval_type = get_eval_type(benchmark_name)
    print(f"Evaluation type: {eval_type}")

    # Check if model supports batch generation (vLLM with continuous batching)
    has_batch = hasattr(model, 'generate_batch') and getattr(model, 'use_vllm', False)

    if has_batch:
        # Prepare all prompts at once, let vLLM handle continuous batching
        print("Preparing all prompts (with ProcessPool)...")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        import os

        # Build prompts in parallel using ThreadPool
        # Video decoding in OpenCV/decord releases GIL, so more workers help
        num_workers = min(128, os.cpu_count() or 32)

        # We need to pass the dataset builder to workers
        # Use indices and rebuild in worker
        indices = list(range(total_samples))

        all_items = [(i, ds.data.iloc[i]) for i in range(total_samples)]
        all_prompts = [None] * total_samples

        # Use ThreadPool with more workers for I/O-bound video loading
        # Video decoding releases GIL in OpenCV/decord, so ThreadPool works
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Check if model supports video input via type='video' (not all models do)
        # Models with VIDEO_LLM=True support type='video', others expect frames as type='image'
        model_supports_video_type = getattr(model, 'VIDEO_LLM', False)

        def build_prompt_for_idx(idx):
            try:
                item = ds.data.iloc[idx]
                # Video benchmarks: use video_llm=True only if model supports type='video'
                if 'vsibench' in benchmark_name.lower() or 'openeqa' in benchmark_name.lower() or 'cosmosreason' in benchmark_name.lower():
                    prompt = ds.build_prompt(item, video_llm=model_supports_video_type)
                elif eval_type == 'pointing':
                    prompt = ds.build_prompt(item, model_name=model_name)
                else:
                    prompt = ds.build_prompt(item)
                return idx, prompt, None
            except Exception as e:
                return idx, None, str(e)

        # Increase workers for video I/O
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(build_prompt_for_idx, i) for i in range(total_samples)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building prompts"):
                idx, prompt, error = future.result()
                if error:
                    print(f"  Error preparing sample {idx}: {error}")
                all_prompts[idx] = prompt

        all_items = [(i, ds.data.iloc[i]) for i in range(total_samples)]

        # Filter valid prompts
        valid_indices = [j for j, p in enumerate(all_prompts) if p is not None]
        valid_prompts = [all_prompts[j] for j in valid_indices]

        # Debug: print first prompt structure (commented out)
        # if valid_prompts:
        #     first_prompt = valid_prompts[0]
        #     types = [item.get('type', 'unknown') for item in first_prompt]
        #     print(f"[DEBUG] First prompt types: {types}")

        # Generate in chunks for progress tracking and error recovery
        # With VLLM_ENABLE_V1_MULTIPROCESSING=0, larger chunks are fine
        print(f"Running inference on {len(valid_prompts)} samples...")

        # Handle empty dataset case
        if len(valid_prompts) == 0:
            print(f"Warning: No valid samples to process for {benchmark_name}")
            return {'benchmark': benchmark_name, 'accuracy': 0.0, 'total': 0, 'correct': 0}

        chunk_size = len(valid_prompts)  # Process all at once; vLLM handles batching internally

        preds = []
        for chunk_start in tqdm(range(0, len(valid_prompts), chunk_size), desc="Inference chunks"):
            chunk_end = min(chunk_start + chunk_size, len(valid_prompts))
            chunk_prompts = valid_prompts[chunk_start:chunk_end]

            try:
                chunk_preds = model.generate_batch(chunk_prompts, dataset=benchmark_name)
                preds.extend(chunk_preds)
            except Exception as e:
                print(f"  Chunk {chunk_start}-{chunk_end} error: {e}")
                preds.extend([''] * len(chunk_prompts))

        # Map predictions back
        pred_map = {valid_indices[j]: preds[j] for j in range(len(preds))}

        # Evaluate
        results = []
        score_sum = 0  # Use score_sum for VSIBench (0-2 per sample), correct count for others
        total = 0
        is_vsibench = (eval_type == 'vsibench')

        for j, (i, item) in enumerate(all_items):
            if item is None:
                results.append({
                    'index': i,
                    'prediction': '',
                    'answer': '',
                    'score': 0,
                    'error': 'Failed to prepare sample',
                })
                total += 1
                continue

            pred = pred_map.get(j, '')

            try:
                result = evaluate_sample(pred, item, eval_type)
                if is_vsibench:
                    score = float(result)  # 0-2 for vsibench
                else:
                    score = 1.0 if result else 0.0
            except Exception as e:
                score = 0.0

            score_sum += score
            total += 1

            results.append({
                'index': i,
                'prediction': pred,
                'answer': str(item.get('answer', '')),
                'score': score,
            })
    else:
        # Sequential processing (fallback for non-vLLM)
        results = []
        score_sum = 0
        total = 0
        is_vsibench = (eval_type == 'vsibench')

        # Check if model supports video input via type='video'
        model_supports_video_type = getattr(model, 'VIDEO_LLM', False)

        for i in tqdm(range(total_samples), desc=f"Processing {benchmark_name}"):
            try:
                item = ds.data.iloc[i]
                # Video benchmarks: use video_llm=True only if model supports type='video'
                if 'vsibench' in benchmark_name.lower() or 'openeqa' in benchmark_name.lower() or 'cosmosreason' in benchmark_name.lower():
                    prompt = ds.build_prompt(item, video_llm=model_supports_video_type)
                elif eval_type == 'pointing':
                    prompt = ds.build_prompt(item, model_name=model_name)
                else:
                    prompt = ds.build_prompt(item)

                pred = model.generate(message=prompt, dataset=benchmark_name)

                result = evaluate_sample(pred, item, eval_type)
                if is_vsibench:
                    score = float(result)
                else:
                    score = 1.0 if result else 0.0

                score_sum += score
                total += 1

                results.append({
                    'index': i,
                    'prediction': pred,
                    'answer': str(item.get('answer', '')),
                    'score': score,
                })

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                results.append({
                    'index': i,
                    'prediction': '',
                    'answer': '',
                    'score': 0.0,
                    'error': str(e),
                })
                total += 1

    # Calculate accuracy/score
    # For VSIBench: average score (MCA: 0-1, NA: 0-2, so max avg ~1.5)
    # For others: percentage correct
    if eval_type == 'vsibench':
        # VSIBench uses MRA which gives 0-2 score for NA questions
        # Report as average score
        avg_score = score_sum / total if total > 0 else 0
        print(f"\nVSIBench Avg Score: {avg_score:.4f} (total samples: {total})")
        accuracy = avg_score * 100  # Scale for consistency with other benchmarks
    else:
        accuracy = score_sum / total * 100 if total > 0 else 0
        print(f"\nAccuracy: {score_sum:.0f}/{total} = {accuracy:.2f}%")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_df = pd.DataFrame(results)
        results_file = output_path / f"{benchmark_name}_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Saved results to: {results_file}")

    return {
        'benchmark': benchmark_name,
        'score_sum': score_sum,
        'total': total,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description='Run full evaluation on embodied benchmarks')
    parser.add_argument('--model', type=str, default='Molmo2-4B',
                        help='Model to evaluate. Supported: Molmo2-4B, Molmo2-8B, LLaVA-OneVision-0.5B, LLaVA-OneVision-7B, Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen3-VL-4B, Phi4-Multimodal, etc.')
    parser.add_argument('--benchmarks', nargs='+', default=None,
                        help='Specific benchmarks to run (default: all)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Max new tokens for generation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Unused - vLLM uses continuous batching automatically')
    args = parser.parse_args()

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model based on model name
    print(f"Initializing {args.model}...")

    # Model configuration mapping
    MODEL_CONFIGS = {
        # Molmo2 models
        'Molmo2-4B': ('molmo2', 'allenai/Molmo2-4B'),
        'Molmo2-8B': ('molmo2', 'allenai/Molmo2-8B'),
        'molmo2-4-spatial-tuning-v1-1k': ('molmo2', '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/mm_olmo/spatial_ckpt/molmo3-spatial/hf_checkpoint'),
        'molmo2-4-spatial-tuning-v1-4k': ('molmo2', '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/mm_olmo/spatial_ckpt/molmo3-spatial/hf_checkpoint_4k'),
        'molmo2-4b-spatial-v3-2k': ('molmo2', '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/converted_models/molmo2-4b-spatial-v3-2k'),
        'molmo2-4b-spatial-v3-7k': ('molmo2', '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/converted_models/molmo2-4b-spatial-v3-7k'),
        # LLaVA-OneVision models (HuggingFace version)
        'LLaVA-OneVision-0.5B': ('llava-onevision', 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'),
        'LLaVA-OneVision-7B': ('llava-onevision', 'llava-hf/llava-onevision-qwen2-7b-ov-hf'),
        'LLaVA-OneVision-72B': ('llava-onevision', 'llava-hf/llava-onevision-qwen2-72b-ov-hf'),
        # Qwen2.5-VL models
        'Qwen2.5-VL-3B': ('qwen2-vl', 'Qwen/Qwen2.5-VL-3B-Instruct'),
        'Qwen2.5-VL-7B': ('qwen2-vl', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        'Qwen2.5-VL-72B': ('qwen2-vl', 'Qwen/Qwen2.5-VL-72B-Instruct'),
        # Qwen3-VL models
        'Qwen3-VL-4B': ('qwen3-vl', 'Qwen/Qwen3-VL-4B-Instruct'),
        'Qwen3-VL-8B': ('qwen3-vl', 'Qwen/Qwen3-VL-8B-Instruct'),
        # Phi4 models
        'Phi4-Multimodal': ('phi4', 'microsoft/Phi-4-multimodal-instruct'),
        # InternVL3 models (matches config.py)
        'InternVL3-1B': ('internvl', 'OpenGVLab/InternVL3-1B'),
        'InternVL3-2B': ('internvl', 'OpenGVLab/InternVL3-2B'),
        'InternVL3-8B': ('internvl', 'OpenGVLab/InternVL3-8B'),
        # InternVL3.5 models (matches config.py, uses underscore in HF path)
        'InternVL3_5-1B': ('internvl', 'OpenGVLab/InternVL3_5-1B'),
        'InternVL3_5-2B': ('internvl', 'OpenGVLab/InternVL3_5-2B'),
        'InternVL3_5-4B': ('internvl', 'OpenGVLab/InternVL3_5-4B'),
        'InternVL3_5-8B': ('internvl', 'OpenGVLab/InternVL3_5-8B'),
    }

    if args.model not in MODEL_CONFIGS:
        print(f"Warning: Model '{args.model}' not in predefined configs.")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        print("Attempting to use model name as HuggingFace path for Molmo2...")
        model_type, model_path = 'molmo2', args.model
    else:
        model_type, model_path = MODEL_CONFIGS[args.model]

    # Instantiate the appropriate model class
    if model_type == 'molmo2':
        model = Molmo2(model_path=model_path, use_vllm=True, max_new_tokens=args.max_new_tokens)
    elif model_type == 'llava-onevision':
        model = LLaVA_OneVision_HF(model_path=model_path, max_new_tokens=args.max_new_tokens)
    elif model_type == 'qwen2-vl':
        model = Qwen2VLChat(model_path=model_path, max_new_tokens=args.max_new_tokens)
    elif model_type == 'qwen3-vl':
        model = Qwen3VLChat(model_path=model_path, max_new_tokens=args.max_new_tokens)
    elif model_type == 'phi4':
        model = Phi4Multimodal(model_path=model_path)
    elif model_type == 'internvl':
        model = InternVLChat(model_path=model_path, version='V2.0', max_new_tokens=args.max_new_tokens)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("Model initialized!")

    # Determine benchmarks to run
    benchmarks = args.benchmarks if args.benchmarks else ALL_BENCHMARKS
    print(f"\nBenchmarks to evaluate: {benchmarks}")

    # Run evaluations
    all_results = []
    start_time = time.time()

    for benchmark in benchmarks:
        result = run_benchmark(model, benchmark, str(output_dir), args.batch_size, model_name=args.model)
        if result:
            all_results.append(result)

    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Benchmark':<30} {'Score/Accuracy':>20}")
    print("-"*50)

    total_score = 0
    total_samples = 0

    for r in all_results:
        score = r.get('score_sum', r.get('correct', 0))  # Backwards compatible
        print(f"{r['benchmark']:<30} {score:>5.1f}/{r['total']:<5} = {r['accuracy']:>6.2f}%")
        total_score += score
        total_samples += r['total']

    overall = total_score / total_samples * 100 if total_samples > 0 else 0
    print("-"*50)
    print(f"{'Overall':<30} {total_score:>5.1f}/{total_samples:<5} = {overall:>6.2f}%")
    print(f"\nTotal time: {elapsed_time/60:.1f} minutes")

    # Save summary
    summary = {
        'model': args.model,
        'timestamp': timestamp,
        'results': all_results,
        'overall_accuracy': overall,
        'total_score': total_score,
        'total_samples': total_samples,
        'elapsed_time_seconds': elapsed_time,
    }

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")


if __name__ == '__main__':
    main()
