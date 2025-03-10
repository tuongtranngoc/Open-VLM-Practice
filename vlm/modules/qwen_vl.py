from tools import config
from vlm.utils.text import make_json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import torch


class QwenDocVQA:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config['ModelConfig']['qwen_vl'],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto").eval()
        
        min_pixels = 1280*28*28
        max_pixels = 5120*28*28
        self.processor = AutoProcessor.from_pretrained(config['ModelConfig']['qwen_vl'],
                                                       min_pixels=min_pixels,
                                                       max_pixels=max_pixels
                                                       )
    
    def generate(self, images, output_format, ocr_token=None):
        system_instruction = open(config['PromptConfig']['system_instruction']).read()
        if ocr_token is None:
            prompt = open(config['PromptConfig']['base_prompt']).read().format(output_format=output_format)
        else:
            prompt = open(config['PromptConfig']['ocrtoken_prompt']).read().format(output_format=output_format, ocr_token=ocr_token)
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        system_instruction += ('\n' + prompt)
        messages[0]['content'].extend([{
            "type": "image",
            "description": f"page-{i+1}"
        } for i in range(len(images))])
        messages[0]['content'].extend([{
            "type": "text",
            "text": system_instruction
        }])

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=3060)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return make_json(output_text[0])