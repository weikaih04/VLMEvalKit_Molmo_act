"""
Full evaluation script for all embodied benchmarks with Molmo2 + vLLM.

Usage:
    python run_full_eval.py --model Molmo2-4B --output results/
    python run_full_eval.py --model Molmo2-4B --benchmarks CVBench_Embodied EmbSpatial_Embodied
    python run_full_eval.py --model Molmo2-4B --batch_size 8 --output results/
"""

import argparse
import os
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

from vlmeval.dataset import build_dataset
from vlmeval.vlm import Molmo2
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
    'MindCube_Embodied',
    'Where2Place_Embodied',
    # 'VSI_Bench_Embodied',  # Temporarily disabled
    'CosmosReason1_Embodied',
    # 'ERQA_Embodied',  # Requires TensorFlow
]


def get_eval_type(benchmark_name: str) -> str:
    """Determine evaluation type for a benchmark."""
    if 'Pointing' in benchmark_name or 'RefSpatial' in benchmark_name or 'Where2Place' in benchmark_name:
        return 'pointing'
    elif 'VSI_Bench' in benchmark_name:
        return 'counting'
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


def evaluate_sample(pred: str, item: dict, eval_type: str) -> bool:
    """Evaluate a single prediction."""
    if eval_type == 'pointing':
        mask = item.get('mask')
        image = item.get('image')
        if image is not None:
            width, height = image.width, image.height
        else:
            width = item.get('image_width', 640)
            height = item.get('image_height', 480)
        return evaluate_pointing(pred, mask, width, height)
    elif eval_type == 'counting':
        return evaluate_counting(pred, str(item.get('answer', '')))
    elif eval_type == 'substring':
        return evaluate_substring(pred, str(item.get('answer', '')))
    else:  # mcq
        return evaluate_mcq(pred, str(item.get('answer', '')))


def run_benchmark(model, benchmark_name: str, output_dir: str = None, batch_size: int = 16):
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
    has_batch = hasattr(model, 'generate_batch') and model.use_vllm

    if has_batch:
        # Prepare all prompts at once, let vLLM handle continuous batching
        print("Preparing all prompts...")
        all_items = []
        all_prompts = []

        for i in tqdm(range(total_samples), desc="Building prompts"):
            try:
                item = ds.data.iloc[i]
                prompt = ds.build_prompt(item)
                all_items.append((i, item))
                all_prompts.append(prompt)
            except Exception as e:
                print(f"  Error preparing sample {i}: {e}")
                all_items.append((i, None))
                all_prompts.append(None)

        # Filter valid prompts
        valid_indices = [j for j, p in enumerate(all_prompts) if p is not None]
        valid_prompts = [all_prompts[j] for j in valid_indices]

        # Generate all at once - vLLM will use continuous batching internally
        print(f"Running inference on {len(valid_prompts)} samples with continuous batching...")
        try:
            preds = model.generate_batch(valid_prompts, dataset=benchmark_name)
        except Exception as e:
            print(f"  Batch generation error: {e}")
            preds = [''] * len(valid_prompts)

        # Map predictions back
        pred_map = {valid_indices[j]: preds[j] for j in range(len(preds))}

        # Evaluate
        results = []
        correct = 0
        total = 0

        for j, (i, item) in enumerate(all_items):
            if item is None:
                results.append({
                    'index': i,
                    'prediction': '',
                    'answer': '',
                    'correct': False,
                    'error': 'Failed to prepare sample',
                })
                total += 1
                continue

            pred = pred_map.get(j, '')

            try:
                is_correct = evaluate_sample(pred, item, eval_type)
            except Exception as e:
                is_correct = False

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'index': i,
                'prediction': pred,
                'answer': str(item.get('answer', '')),
                'correct': is_correct,
            })
    else:
        # Sequential processing (fallback for non-vLLM)
        results = []
        correct = 0
        total = 0

        for i in tqdm(range(total_samples), desc=f"Processing {benchmark_name}"):
            try:
                item = ds.data.iloc[i]
                prompt = ds.build_prompt(item)

                pred = model.generate(message=prompt, dataset=benchmark_name)

                is_correct = evaluate_sample(pred, item, eval_type)

                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    'index': i,
                    'prediction': pred,
                    'answer': str(item.get('answer', '')),
                    'correct': is_correct,
                })

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                results.append({
                    'index': i,
                    'prediction': '',
                    'answer': '',
                    'correct': False,
                    'error': str(e),
                })
                total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")

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
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description='Run full evaluation on embodied benchmarks')
    parser.add_argument('--model', type=str, default='Molmo2-4B',
                        choices=['Molmo2-4B', 'Molmo2-8B'],
                        help='Model to evaluate')
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

    # Initialize model
    print(f"Initializing {args.model} with vLLM...")
    model_path = f"allenai/{args.model}"
    model = Molmo2(model_path=model_path, use_vllm=True, max_new_tokens=args.max_new_tokens)
    print("Model initialized!")

    # Determine benchmarks to run
    benchmarks = args.benchmarks if args.benchmarks else ALL_BENCHMARKS
    print(f"\nBenchmarks to evaluate: {benchmarks}")

    # Run evaluations
    all_results = []
    start_time = time.time()

    for benchmark in benchmarks:
        result = run_benchmark(model, benchmark, str(output_dir), args.batch_size)
        if result:
            all_results.append(result)

    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Benchmark':<30} {'Accuracy':>20}")
    print("-"*50)

    total_correct = 0
    total_samples = 0

    for r in all_results:
        print(f"{r['benchmark']:<30} {r['correct']:>5}/{r['total']:<5} = {r['accuracy']:>6.2f}%")
        total_correct += r['correct']
        total_samples += r['total']

    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print("-"*50)
    print(f"{'Overall':<30} {total_correct:>5}/{total_samples:<5} = {overall:>6.2f}%")
    print(f"\nTotal time: {elapsed_time/60:.1f} minutes")

    # Save summary
    summary = {
        'model': args.model,
        'timestamp': timestamp,
        'results': all_results,
        'overall_accuracy': overall,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'elapsed_time_seconds': elapsed_time,
    }

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")


if __name__ == '__main__':
    main()
