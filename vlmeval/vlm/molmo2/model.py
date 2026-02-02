"""
Molmo2 VLMEvalKit Integration

Supports both transformers and vLLM backends for Molmo2 models.
vLLM backend uses the mm_olmo plugin for efficient inference.
"""

import os
import sys
import torch
from PIL import Image
from ..base import BaseModel
from ...smp import *


class Molmo2(BaseModel):
    """
    Molmo2 model wrapper for VLMEvalKit.

    Supports:
    - Transformers backend (default)
    - vLLM backend (use_vllm=True) - requires mm_olmo
    - Multiple images
    - Video frames (as multiple images)
    """

    INSTALL_REQ = False
    INTERLEAVE = True  # Support interleaved image/text

    def __init__(
        self,
        model_path='allenai/Molmo2-4B',
        use_vllm=False,
        mm_olmo_path='/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/mm_olmo',
        **kwargs
    ):
        """
        Initialize Molmo2 model.

        Args:
            model_path: HuggingFace model ID or local path
            use_vllm: Whether to use vLLM backend (faster)
            mm_olmo_path: Path to mm_olmo directory (for vLLM plugin)
            **kwargs: Additional arguments
                - max_new_tokens: Maximum tokens to generate (default: 200)
                - tensor_parallel_size: GPU count for vLLM (default: auto)
                - gpu_memory_utilization: GPU memory fraction (default: 0.9)
        """
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.mm_olmo_path = mm_olmo_path
        self.max_new_tokens = kwargs.get('max_new_tokens', 200)
        self.kwargs = kwargs

        if self.use_vllm:
            self._init_vllm()
        else:
            self._init_transformers()

    def _init_vllm(self):
        """Initialize vLLM backend using mm_olmo plugin."""
        # Set environment variables
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['VLLM_USE_V1'] = '0'

        # Add mm_olmo to path
        if self.mm_olmo_path not in sys.path:
            sys.path.insert(0, self.mm_olmo_path)

        try:
            from vllm import LLM, SamplingParams, ModelRegistry
            from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
            from olmo.vllm.molmo2 import Molmo2ForConditionalGeneration
            from transformers import AutoProcessor

            # Register Molmo2 with vLLM
            try:
                ModelRegistry.register_model(
                    "Molmo2ForConditionalGeneration",
                    Molmo2ForConditionalGeneration
                )
                _MULTIMODAL_MODELS["Molmo2ForConditionalGeneration"] = (
                    "molmo2", "Molmo2ForConditionalGeneration"
                )
            except Exception:
                # Model already registered
                pass

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            # Initialize vLLM
            # Note: tensor parallelism with mm_olmo plugin has issues with model registration
            # in worker processes. Force tp=1 for now, continuous batching still provides speedup.
            tensor_parallel_size = self.kwargs.get('tensor_parallel_size', 1)
            gpu_memory_utilization = self.kwargs.get('gpu_memory_utilization', 0.9)
            max_model_len = self.kwargs.get('max_model_len', 32768)
            max_images = self.kwargs.get('max_images_per_prompt', 16)

            max_num_seqs = self.kwargs.get('max_num_seqs', 16)  # 并发序列数

            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                limit_mm_per_prompt={"image": max_images, "video": 1},
            )

            self.sampling_params_class = SamplingParams
            print(f"Molmo2 vLLM initialized: {self.model_path}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import vLLM or mm_olmo components. "
                f"Make sure mm_olmo is at {self.mm_olmo_path} and vLLM is installed. "
                f"Error: {e}"
            )

    def _init_transformers(self):
        """Initialize transformers backend."""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto"
        )

        print(f"Molmo2 transformers initialized: {self.model_path}")

    def generate_inner(self, message, dataset=None):
        """
        Generate response for the given message.

        Args:
            message: List of dicts with 'type' and 'value' keys
                     type='image' -> value is image path
                     type='text' -> value is text string
            dataset: Dataset name (optional)

        Returns:
            Generated text response
        """
        if self.use_vllm:
            return self._generate_vllm(message, dataset)
        else:
            return self._generate_transformers(message, dataset)

    def _generate_transformers(self, message, dataset=None):
        """Generate using transformers backend."""
        # Build content list for Molmo2 format
        content = []
        for item in message:
            if item['type'] == 'image':
                # Handle both path and PIL Image
                if isinstance(item['value'], str):
                    img = Image.open(item['value']).convert('RGB')
                elif isinstance(item['value'], Image.Image):
                    img = item['value'].convert('RGB')
                else:
                    continue
                content.append({"type": "image", "image": img})
            elif item['type'] == 'text':
                content.append({"type": "text", "text": item['value']})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=self.max_new_tokens
            )

        # Decode only the generated part
        gen_text = self.processor.tokenizer.decode(
            output[0, inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )

        return gen_text.strip()

    def _prepare_vllm_input(self, message):
        """Prepare a single vLLM input from a message."""
        content = []
        images = []

        for item in message:
            if item['type'] == 'image':
                # Handle both path and PIL Image
                if isinstance(item['value'], str):
                    img = Image.open(item['value']).convert('RGB')
                elif isinstance(item['value'], Image.Image):
                    img = item['value'].convert('RGB')
                else:
                    continue
                images.append(img)
                content.append({"type": "image", "image": img})
            elif item['type'] == 'text':
                content.append({"type": "text", "text": item['value']})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare vLLM input
        mm_data = {}
        if images:
            mm_data['image'] = images

        return {
            'prompt': prompt,
            'multi_modal_data': mm_data,
        }

    def _generate_vllm(self, message, dataset=None):
        """Generate using vLLM backend (single input)."""
        vllm_input = self._prepare_vllm_input(message)

        # Sampling parameters
        sampling_params = self.sampling_params_class(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )

        # Generate
        outputs = self.llm.generate([vllm_input], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text

        # Handle </think> tag if present
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        return response.strip()

    def generate_batch(self, messages, dataset=None):
        """
        Generate responses for a batch of messages using vLLM.

        Args:
            messages: List of messages, each message is a list of dicts
            dataset: Dataset name (optional)

        Returns:
            List of generated text responses
        """
        if not self.use_vllm:
            # Fallback to sequential generation for transformers
            return [self._generate_transformers(msg, dataset) for msg in messages]

        # Prepare all inputs in parallel using ThreadPool
        from concurrent.futures import ThreadPoolExecutor
        import os

        num_workers = min(32, os.cpu_count() or 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            vllm_inputs = list(executor.map(self._prepare_vllm_input, messages))

        # Sampling parameters
        sampling_params = self.sampling_params_class(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )

        # Generate batch
        outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)

        # Extract responses
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            # Handle </think> tag if present
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            responses.append(response.strip())

        return responses
