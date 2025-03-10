from collections import defaultdict
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.ocr import run_ocr
from PIL import Image

import numpy as np
import json
import cv2
import os


class OcrExtractor:
    def __init__(self) :
        self.langs = ['en', 'vi']
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        self.thresh_rec = 0.8
    
    def extract(self, images):
        ocr_token = defaultdict(list)
        predictions = run_ocr(images, [self.langs for __ in range(len(images))], self.det_model, self.det_processor, self.rec_model, self.rec_processor, batch_size=160)
        if len(predictions) > 0:
            for i, __ in enumerate(images):
                texts = []
                boxes = []
                scores = []
                for text_line in predictions[i].text_lines:
                    if text_line.confidence >= self.thresh_rec:
                        texts.append(text_line.text)
                        boxes.append(text_line.bbox)
                        scores.append(round(text_line.confidence, 3))
                        ocr_token[f"page_{i+1}"].append({
                            "text": text_line.text,
                            "bbox": text_line.bbox,
                            "confidence": round(text_line.confidence, 3)
                        })
        return texts, boxes, scores, ocr_token
    
    def debug(self, basename, image, texts, boxes, scores, save_dir):     
        os.makedirs(save_dir, exist_ok=True)
        for text, bbox, conf in zip(texts, boxes, scores):
            image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                                  color=(0, 0, 255),
                                  thickness=1)

            image = cv2.putText(image, f'{round(conf, 3)}', (int(bbox[2]), int(bbox[1])-10), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.7,
                                thickness=2,
                                color=(0, 0, 255))
        
        cv2.imwrite(os.path.join(save_dir, basename), image)
        
    def save_image(self, basename, image, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, basename), image)



if __name__ == "__main__":
    ocr = OcrExtractor()
    img1_path = '/home/tuongtn/workspace/Projects/Shinhan_OCR/data/shinhan_ocr_dpi500/1.Contract/Template04_2pages_2.png'
    basename = os.path.basename(img1_path)
    img1 = Image.open(img1_path)
    img_pil = img1.convert('RGB')
    img_rgb = np.array(img_pil)[:, :, ::-1].copy()
    texts, boxes, scores, ocr_token = ocr.extract([img1])
    json.dump(ocr_token, open(f'.results/{os.path.splitext(basename)[0]}_ocr.json', 'w'), ensure_ascii=False)
    ocr.debug(basename, img_rgb, texts, boxes, scores, '.debug')