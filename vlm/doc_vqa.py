import time
import concurrent.futures
from collections import defaultdict

from vlm.utils.image import *
from vlm.utils.matching import *
from vlm.modules.qwen_vl import QwenDocVQA
from vlm.modules.gemini import GeminiDocVQA
from traditional_ocr.ocr_surya import OcrExtractor

from tools import config as cfg


class QwenDocExtractor:
    def __init__(self):
        self.doc_vqa = QwenDocVQA()
        self.doc_ocr = OcrExtractor()
    
    def extract_wt_logic(self, pdf, output_format, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
            
        images_for_ocr = copy.deepcopy(images)
        images_for_vqa = copy.deepcopy(images)

        st = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(self.doc_ocr.extract, images_for_ocr)
            vqa_future = executor.submit(self.doc_vqa.generate, images_for_vqa, output_format)
            
            ocr_results = ocr_future.result()[-1]
            vqa_results = vqa_future.result()
        
        final_res = assign_confidence(vqa_results, ocr_results)
        total_time = time.time() - st

        return final_res, total_time
    
    def extract_wt_prompt(self, pdf, prompt, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
        vqa_results = self.doc_vqa.generate_wt_prompt(images, prompt)
        return vqa_results
    

class GeminiDocExtractor:
    def __init__(self):
        self.doc_vqa = GeminiDocVQA()
        self.doc_ocr = OcrExtractor()

    def extract_wt_ocrtoken(self, pdf, output_format, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
            
        ocr_st = time.time()
        ocr_results = self.doc_ocr.extract(images)[-1]
        ocr_time = time.time() - ocr_st
        vqa_st = time.time()
        vqa_results = self.doc_vqa.generate(images, output_format, ocr_token=dict(ocr_results))
        vqa_time = time.time() - vqa_st
        return vqa_results, ocr_time, vqa_time
    
    def extract_logic(self, pdf, output_format, pdf_type='from_path'):
        if pdf_type == 'from_path':
            images = pdf_to_images_from_path(pdf)[:cfg['Dataset']['max_page']]
        elif pdf_type == 'from_bytes':
            images = pdf_to_images_from_bytes(pdf.read())[:cfg['Dataset']['max_page']]
        else:
            images = [read_byte_io(pdf)]
        
        images_for_ocr = copy.deepcopy(images)
        images_for_vqa = copy.deepcopy(images)

        st = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(self.doc_ocr.extract, images_for_ocr)
            vqa_future = executor.submit(self.doc_vqa.generate, images_for_vqa, output_format)
            
            ocr_results = ocr_future.result()[-1]
            vqa_results = vqa_future.result()
        
        final_res = assign_confidence(vqa_results, ocr_results)
        total_time = time.time() - st

        return final_res, total_time