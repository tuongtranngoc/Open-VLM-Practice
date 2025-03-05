import time
from PIL import Image

from vlm.utils.image import *
from vlm.modules.qwen_vl import DocVQA

from tools import config as cfg


class DocExtractor:
    def __init__(self):
        self.doc_vqa = DocVQA()
    
    def extract_wt_outformat(self, pdf, output_format, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
        vqa_results = self.doc_vqa.extract_wt_outformat(images, output_format)
        return vqa_results
    
    def extract_wt_prompt(self, pdf, prompt, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
        vqa_results = self.doc_vqa.extract_wt_prompt(images, prompt)
        return vqa_results
    
    def extract_wt_ocrtoken(self, pdf, output_format, ocr_token, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
        vqa_results = self.doc_vqa.extract_wt_ocrtoken(images, output_format, ocr_token)
        return vqa_results


if __name__ == "__main__":
    import time
    DATA_PATH = '/home/tuongtran/tuongtn/Researching/Researching_Projects/DocumentAI/Open-VLM-Practice/data/shinhan_ocr_500/1.Contract/Template08_1page/page_1.png'
    doc_extract = DocExtractor()
    st = time.time()
    ocr_token = open('src/modules/ocr/.debug/ocr_token.txt').read()
    output_format = open('prompts/output_format.txt').read()
    res = doc_extract.extract_wt_ocrtoken(DATA_PATH, output_format, ocr_token, pdf_type='png')
    print(res)
    print(f"Processing time: {round(time.time()-st, 3)} (s)")