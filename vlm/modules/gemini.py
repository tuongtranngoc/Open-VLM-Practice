import json
import os
from google import genai
from google.genai import types

from tools import *
from vlm.utils.image import *
from vlm.utils.text import make_json


class GeminiDocVQA:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.model_id = config['ModelConfig']['google_gemini']
    
    def generate(self, images, output_format, ocr_token=None):
        # for i in range(len(images)):
        #     images[i].thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        system_instruction = open(config['PromptConfig']['system_instruction']).read()
        if ocr_token is None:
            prompt = open(config['PromptConfig']['base_prompt']).read().format(output_format=output_format)
        else:
            prompt = open(config['PromptConfig']['ocrtoken_prompt']).read().format(output_format=output_format, ocr_token=ocr_token)
        
        if len(images) == 1:
            page_note = f"Note: There is a page in document"
        elif len(images) > 1:
            page_note = f"Note: There are {len(images)} pages in document"
        else:
            page_note = ""
        prompt += ('\n' + page_note)
        contents = [prompt, images]
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5,
                top_p=0.95,
                max_output_tokens=12000,
                candidate_count=1,
            )
        )
        return make_json(response.candidates[0].content.parts[0].text)


if __name__ == "__main__":
    import time
    doc_vqp = GeminiDocVQA()
    img1 = '/home/tuongtn/workspace/Projects/Shinhan_OCR/vlm/modules/Template08_1page_1_1.png'
    img2 = '/home/tuongtn/workspace/Projects/Shinhan_OCR/vlm/modules/Template08_1page_1_2.png'
    img = "/home/tuongtn/workspace/Projects/Shinhan_OCR/data/shinhan_ocr_dpi500/1.Contract/Template08_1page_1.png"
    images = [
        # Image.open(img1),
        # Image.open(img2),
        Image.open(img),
    ]
    output_format = open('prompts/output_format.txt').read()
    st = time.time()
    res = doc_vqp.generate(images, output_format)
    print(res)
    print(time.time() - st)