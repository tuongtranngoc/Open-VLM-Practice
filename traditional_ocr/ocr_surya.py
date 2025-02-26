from surya.ocr import run_ocr
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from PIL import Image

import numpy as np
import cv2
import os


class OcrExtractor:
    def __init__(self):
        self.langs = ['km', 'en', 'vi']
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        self.thresh_rec = 0.2

    def extract(self, image):
        texts = []
        boxes = []
        scores = []
        predictions = run_ocr([image], [self.langs], self.det_model, self.det_processor, self.rec_model, self.rec_processor)
        if len(predictions) > 0:
            for text_line in predictions[0].text_lines:
                if text_line.confidence >= self.thresh_rec:
                    texts.append(text_line.text)
                    boxes.append(text_line.bbox)
                    scores.append(text_line.confidence)
        return texts, boxes, scores

    def debug(self, basename, image, texts, boxes, scores, save_dir):     
        os.makedirs(save_dir, exist_ok=True)
        for text, bbox, conf in zip(texts, boxes, scores):
            image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                                  color=(0, 0, 255),
                                  thickness=2)
            image = cv2.putText(image, f'{round(conf, 3)}', (int(bbox[2]), int(bbox[1])-10), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.3,
                                thickness=1,
                                color=(0, 0, 255))
        print(texts)
        cv2.imwrite(os.path.join(save_dir, basename), image)


if __name__ == "__main__":
    ocr = OcrExtractor()
    img_path = ''
    basename = os.path.basename(img_path)
    img = Image.open(img_path)
    texts, boxes, scores = ocr.extract(img)
    img_pil = img.convert('RGB')
    img_rgb = np.array(img_pil)[:, :, ::-1].copy()
    os.makedirs('.results', exist_ok=True)
    open('.results/ocr_token.txt', 'w').write(str(texts))
    ocr.debug(basename, img_rgb, texts, boxes, scores, '.debug')