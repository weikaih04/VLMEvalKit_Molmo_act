"""
Test all 11 embodied benchmarks with Molmo2-4B.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from vlmeval.dataset import build_dataset
from vlmeval.vlm import Molmo2
from vlmeval.dataset.embodied_benchmarks.utils.point_extraction import (
    extract_molmo_points, extract_points_robust, check_point_in_mask
)

# All 11 benchmarks to test
BENCHMARKS = [
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

NUM_SAMPLES = 10  # Test 10 samples per benchmark


def evaluate_mcq(pred: str, answer: str) -> bool:
    """Evaluate multiple choice question using robust answer extraction."""
    import re
    import string

    answer = answer.strip()
    pred = pred.strip()

    # Check if answer is a number (e.g., VSI_Bench)
    if answer.isdigit():
        # Extract first number from prediction
        pred_match = re.search(r'\d+', pred)
        if pred_match:
            return pred_match.group(0) == answer
        return False

    # Use robust extraction for letters (matching notebook behavior)
    def extract_letter(text, max_letter='H'):
        """Extract answer letter with priority patterns."""
        if not text:
            return None
        # Convert to uppercase first (matching notebook behavior)
        text = text.strip().upper()
        valid_letters = string.ascii_uppercase[:ord(max_letter) - ord('A') + 1]
        letter_pattern = f'[{valid_letters}]'

        patterns = [
            rf'\(({letter_pattern})\)',  # (A)
            rf'ANSWER:?\s*IS?\s*\(?({letter_pattern})\)?',  # ANSWER: A
            rf'^({letter_pattern})\.',  # A.
            rf'^({letter_pattern})\)',  # A)
            rf'^({letter_pattern}):',  # A:
            rf'^({letter_pattern})\s',  # A (at start)
            rf'\s({letter_pattern})$',  # A (at end)
            rf'\b({letter_pattern})\b',  # standalone A
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        if len(text) == 1 and text in valid_letters:
            return text
        return None

    pred_letter = extract_letter(pred)
    answer_letter = extract_letter(answer)

    if not pred_letter or not answer_letter:
        return answer.lower() in pred.lower()

    return pred_letter == answer_letter


def evaluate_pointing(pred: str, mask, width: int, height: int) -> bool:
    """Evaluate pointing task - any point in mask."""
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


def evaluate_substring(pred: str, answer: str) -> bool:
    """Evaluate by substring matching."""
    pred_lower = pred.lower().strip()
    answer_lower = answer.lower().strip()
    return answer_lower in pred_lower


def evaluate_counting(pred: str, answer: str) -> bool:
    """Evaluate counting task by counting points in prediction."""
    import re

    try:
        answer_num = int(answer.strip())
    except ValueError:
        return False

    # Count points in coords="..." format
    coords_match = re.search(r'coords="([^"]+)"', pred)
    if coords_match:
        coords_str = coords_match.group(1)
        # Count by semicolons (each point separated by ;)
        if ';' in coords_str:
            num_points = coords_str.count(';') + 1
        else:
            # Count by point patterns: "prefix x y"
            num_points = len(re.findall(r'[0-9]+ [0-9]{3,4} [0-9]{3,4}', coords_str))
        return num_points == answer_num

    # Fallback: try to extract first number from prediction
    pred_match = re.search(r'\d+', pred)
    if pred_match:
        return int(pred_match.group(0)) == answer_num

    return False


def test_benchmark(model, benchmark_name: str, num_samples: int = 10):
    """Test a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Testing: {benchmark_name}")
    print(f"{'='*60}")

    try:
        ds = build_dataset(benchmark_name)
        total_samples = len(ds.data)
        print(f"Total samples: {total_samples}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    num_test = min(num_samples, total_samples)
    correct = 0
    total = 0

    # Determine evaluation type
    is_pointing = 'Pointing' in benchmark_name or 'RefSpatial' in benchmark_name or 'Where2Place' in benchmark_name
    is_vqa_substring = 'RoboSpatial_VQA' in benchmark_name
    is_counting = 'VSI_Bench' in benchmark_name

    for i in range(num_test):
        try:
            item = ds.data.iloc[i]
            prompt = ds.build_prompt(item)

            pred = model.generate(message=prompt, dataset=benchmark_name)

            if is_pointing:
                # Pointing evaluation
                mask = item.get('mask')
                image = item.get('image')
                if image is not None:
                    width, height = image.width, image.height
                else:
                    width, height = item.get('image_width', 640), item.get('image_height', 480)

                is_correct = evaluate_pointing(pred, mask, width, height)
            elif is_counting:
                # Counting evaluation (VSI_Bench)
                answer = str(item.get('answer', ''))
                is_correct = evaluate_counting(pred, answer)
            elif is_vqa_substring:
                # VQA substring matching
                answer = str(item.get('answer', ''))
                is_correct = evaluate_substring(pred, answer)
            else:
                # MCQ evaluation
                answer = str(item.get('answer', ''))
                is_correct = evaluate_mcq(pred, answer)

            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:2d}/{num_test}] {status} | Pred: {pred[:50]}...")

        except Exception as e:
            print(f"  [{i+1:2d}/{num_test}] Error: {e}")
            total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1f}%")

    return {
        'benchmark': benchmark_name,
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
    }


def main():
    print("Initializing Molmo2-4B with vLLM...")
    model = Molmo2(model_path='allenai/Molmo2-4B', use_vllm=True, max_new_tokens=512)
    print("Model initialized!\n")

    results = []

    for benchmark in BENCHMARKS:
        result = test_benchmark(model, benchmark, NUM_SAMPLES)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Benchmark':<30} {'Accuracy':>15}")
    print("-"*45)

    total_correct = 0
    total_samples = 0

    for r in results:
        print(f"{r['benchmark']:<30} {r['correct']:>3}/{r['total']:<3} = {r['accuracy']:>5.1f}%")
        total_correct += r['correct']
        total_samples += r['total']

    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    print("-"*45)
    print(f"{'Overall':<30} {total_correct:>3}/{total_samples:<3} = {overall:>5.1f}%")


if __name__ == '__main__':
    main()
