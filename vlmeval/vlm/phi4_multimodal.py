from PIL import Image
import torch

from .base import BaseModel
from ..smp import *


class Phi4Multimodal(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='microsoft/Phi-4-multimodal-instruct', **kwargs):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        # Monkey-patch peft to fix compatibility with Phi4's internal LoRA
        # Phi4 calls get_peft_model on its inner model which lacks prepare_inputs_for_generation
        import peft.peft_model as _peft_module
        _orig_causal_init = _peft_module.PeftModelForCausalLM.__init__

        def _patched_causal_init(self_peft, model, peft_config, adapter_name='default', **kw):
            if not hasattr(model, 'prepare_inputs_for_generation'):
                model.prepare_inputs_for_generation = lambda *a, **k: None
            _orig_causal_init(self_peft, model, peft_config, adapter_name, **kw)

        _peft_module.PeftModelForCausalLM.__init__ = _patched_causal_init

        # Try flash_attention_2 first, fall back to eager if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map='cuda', trust_remote_code=True,
                torch_dtype='auto', attn_implementation='flash_attention_2'
            ).eval()
        except (ImportError, Exception) as e:
            logging.warning(f"flash_attn not available ({e}), falling back to eager attention")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config._attn_implementation = 'eager'
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, device_map='cuda', trust_remote_code=True,
                torch_dtype='auto', attn_implementation='eager'
            ).eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model_path)

        self.model = model
        self.processor = processor
        # self.kwargs = kwargs
        self.generation_config = generation_config

    def generate_inner(self, message, dataset=None):
        user_question = '\n'.join([msg['value'] for msg in message if msg['type'] == 'text'])
        images = [Image.open(msg['value']).convert('RGB') for msg in message if msg['type'] == 'image']

        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        prompt = f'{user_prompt}<|image_placeholder|>{user_question}{prompt_suffix}{assistant_prompt}'
        image_prompt = ''
        for num in range(1, len(images) + 1):
            image_prompt += f'<|image_{num}|>'
        prompt = prompt.replace('<|image_placeholder|>', image_prompt, 1)

        inputs = self.processor(text=prompt, images=images, return_tensors='pt').to('cuda')

        # Generate response
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
