#!/usr/bin/env python3
"""Debug script to test VSIBench video loading with only 10 samples."""

import os
import sys

# Add paths
sys.path.insert(0, '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/VLMEvalKit_embodied_benchmark')
sys.path.insert(0, '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/mm_olmo')

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

def test_build_prompt():
    """Test VSIBench build_prompt with video_llm=True"""
    print("=" * 60)
    print("Step 1: Testing VSIBench build_prompt")
    print("=" * 60)

    from vlmeval.dataset.embodied_benchmarks.vsi_bench import VSIBenchDataset

    ds = VSIBenchDataset(dataset='vsibench_32frame')
    print(f"Total samples: {len(ds)}")

    # Test first sample
    item = ds.data.iloc[0]
    print(f"\nSample 0 video_path: {item['video_path']}")
    print(f"Video exists: {os.path.exists(item['video_path'])}")

    # Build prompt with video_llm=True
    prompt = ds.build_prompt(item, video_llm=True)
    types = [p.get('type', 'unknown') for p in prompt]
    print(f"\nPrompt types with video_llm=True: {types}")

    # Build prompt with video_llm=False (default)
    prompt_default = ds.build_prompt(item, video_llm=False)
    types_default = [p.get('type', 'unknown') for p in prompt_default]
    print(f"Prompt types with video_llm=False: {types_default}")

    return prompt


def test_prepare_vllm_input(prompt):
    """Test _prepare_vllm_input with the prompt"""
    print("\n" + "=" * 60)
    print("Step 2: Testing _prepare_vllm_input")
    print("=" * 60)

    from vlmeval.vlm.molmo2.model import Molmo2
    from PIL import Image

    # We need to test the _prepare_vllm_input logic without loading the full model
    # Let's manually test the video loading part

    print("\nPrompt content:")
    for i, item in enumerate(prompt):
        print(f"  [{i}] type={item.get('type')}, value={str(item.get('value'))[:100]}...")

    # Check if video type exists
    video_items = [p for p in prompt if p.get('type') == 'video']
    image_items = [p for p in prompt if p.get('type') == 'image']

    print(f"\nVideo items: {len(video_items)}")
    print(f"Image items: {len(image_items)}")

    if video_items:
        video_path = video_items[0]['value']
        print(f"\nTesting video loading for: {video_path}")

        # Test video loading
        from olmo.vllm.video_backend import Molmo2VideoBackend

        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        print(f"Video file size: {len(video_bytes) / 1024 / 1024:.2f} MB")

        frames, metadata = Molmo2VideoBackend.load_bytes(
            video_bytes,
            backend="decord",
            frame_sample_mode="uniform_last_frame",
            num_frames=32,
            max_fps=8,
        )

        print(f"Loaded frames shape: {frames.shape}")
        print(f"Metadata: {metadata}")

        return (frames, metadata)

    return None


def test_vllm_mm_data_parsing(video_frames):
    """Test how vLLM parses mm_data"""
    print("\n" + "=" * 60)
    print("Step 3: Testing vLLM mm_data parsing")
    print("=" * 60)

    from vllm.multimodal.parse import MultiModalDataParser

    # Create mm_data as our code does
    mm_data = {
        'video': [video_frames]  # [(frames_array, metadata)]
    }

    print(f"\nmm_data keys: {list(mm_data.keys())}")
    print(f"mm_data['video'] length: {len(mm_data['video'])}")
    print(f"mm_data['video'][0] type: {type(mm_data['video'][0])}")
    if isinstance(mm_data['video'][0], tuple):
        frames, meta = mm_data['video'][0]
        print(f"  frames shape: {frames.shape}")
        print(f"  metadata keys: {list(meta.keys())}")

    # Parse with vLLM parser (with video_needs_metadata=True like mm_olmo)
    parser = MultiModalDataParser(video_needs_metadata=True)

    try:
        mm_items = parser.parse_mm_data(mm_data)
        print(f"\nParsed mm_items keys: {list(mm_items.keys())}")
        for modality, items in mm_items.items():
            print(f"  {modality}: count={len(items)}, type={type(items).__name__}")
    except Exception as e:
        print(f"\nError parsing mm_data: {e}")
        import traceback
        traceback.print_exc()


def test_full_inference():
    """Test full inference with 1 sample"""
    print("\n" + "=" * 60)
    print("Step 4: Testing full inference (1 sample)")
    print("=" * 60)

    from vlmeval.vlm.molmo2.model import Molmo2
    from vlmeval.dataset.embodied_benchmarks.vsi_bench import VSIBenchDataset

    # Load dataset
    ds = VSIBenchDataset(dataset='vsibench_32frame')

    # Load model with vLLM
    print("\nLoading Molmo2-4B model with vLLM...")
    model = Molmo2(
        model_path='allenai/Molmo2-4B',
        use_vllm=True,
        max_new_tokens=512,
        max_video_frames=32,
    )

    # Get first sample
    item = ds.data.iloc[0]
    prompt = ds.build_prompt(item, video_llm=True)

    print(f"\nRunning inference on 1 sample...")
    print(f"Prompt types: {[p.get('type') for p in prompt]}")

    try:
        response = model.generate_inner(prompt, dataset='vsibench_32frame')
        print(f"\nResponse: {response[:200]}...")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()


def test_batch_inference(batch_size=5):
    """Test batch inference with multiple samples"""
    print("\n" + "=" * 60)
    print(f"Step 5: Testing batch inference ({batch_size} samples)")
    print("=" * 60)

    from vlmeval.vlm.molmo2.model import Molmo2
    from vlmeval.dataset.embodied_benchmarks.vsi_bench import VSIBenchDataset

    # Load dataset
    ds = VSIBenchDataset(dataset='vsibench_32frame')

    # Load model with vLLM
    print("\nLoading Molmo2-4B model with vLLM...")
    model = Molmo2(
        model_path='allenai/Molmo2-4B',
        use_vllm=True,
        max_new_tokens=512,
        max_video_frames=32,
    )

    # Get batch of samples
    prompts = []
    for i in range(min(batch_size, len(ds))):
        item = ds.data.iloc[i]
        prompt = ds.build_prompt(item, video_llm=True)
        prompts.append(prompt)
        print(f"Sample {i}: types={[p.get('type') for p in prompt]}")

    print(f"\nRunning batch inference on {len(prompts)} samples...")

    try:
        responses = model.generate_batch(prompts, dataset='vsibench_32frame')
        print(f"\nGot {len(responses)} responses:")
        for i, resp in enumerate(responses):
            print(f"  [{i}] {resp[:100]}...")
    except Exception as e:
        print(f"\nError during batch inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0, help='Which step to run (0=all, 1-5=specific)')
    parser.add_argument('--batch', type=int, default=5, help='Batch size for step 5')
    args = parser.parse_args()

    if args.step == 0 or args.step == 1:
        prompt = test_build_prompt()

    if args.step == 0 or args.step == 2:
        if args.step == 2:
            prompt = test_build_prompt()
        video_frames = test_prepare_vllm_input(prompt)

    if args.step == 0 or args.step == 3:
        if args.step == 3:
            prompt = test_build_prompt()
            video_frames = test_prepare_vllm_input(prompt)
        if video_frames:
            test_vllm_mm_data_parsing(video_frames)

    if args.step == 4:
        test_full_inference()

    if args.step == 5:
        test_batch_inference(args.batch)

    print("\n" + "=" * 60)
    print("Debug test completed!")
    print("=" * 60)
