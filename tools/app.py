import time
from flask_cors import CORS
from flask import Flask, request

from vlm.doc_vqa import QwenDocExtractor, GeminiDocExtractor
from tools import config as cfg


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

qwen_doc_extractor = QwenDocExtractor()
gemini_doc_extractor = GeminiDocExtractor()


@app.route('/api/v1/qwen-doc-vqa', methods=['POST'])
def doc_vqa_v1():
    if 'pdf' not in request.files:
        return {
            'status': 'error',
            'message': 'Not found pdf file'
        }
    
    pdf_file = request.files['pdf']
    output_format = request.form.get('output_format', None)
    
    
    if output_format is None:
        return {
            'status': 'error',
            'message': 'Not found output format',
        }
    try:
        result, total_time = qwen_doc_extractor.extract_wt_logic(pdf=pdf_file,
                                       output_format=output_format,
                                        pdf_type='from_bytes')
        return {
            'status': 'success',
            'message': result,
            'time': total_time
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
    

@app.route('/api/v1/gemini-doc-vqa', methods=['POST'])
def gemini_doc_vqa_v1():
    if 'pdf' not in request.files:
        return {
            'status': 'error',
            'message': 'Not found pdf file'
        }
    
    pdf_file = request.files['pdf']
    output_format = request.form.get('output_format', None)
    
    if output_format is None:
        return {
            'status': 'error',
            'message': 'Not found output format',
        }
    try:
        result, ocr_time, vqa_time = gemini_doc_extractor.extract(pdf=pdf_file,
                                       output_format=output_format,
                                        pdf_type="from_bytes")
        return {
            'status': 'success',
            'message': result,
            'ocr_time': ocr_time,
            'vqa_time': vqa_time,
            'total_time': ocr_time + vqa_time,
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
        
        
@app.route('/api/v2/gemini-doc-vqa', methods=['POST'])
def gemini_doc_vqa_v2():
    if 'pdf' not in request.files:
        return {
            'status': 'error',
            'message': 'Not found pdf file'
        }
    
    pdf_file = request.files['pdf']
    output_format = request.form.get('output_format', None)
    
    if output_format is None:
        return {
            'status': 'error',
            'message': 'Not found output format',
        }
    try:
        result, total_time = gemini_doc_extractor.extract_logic(pdf=pdf_file,
                                       output_format=output_format,
                                        pdf_type="from_bytes")
        return {
            'status': 'success',
            'message': result,
            'total_time': total_time,
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=cfg['Deploy']['port'], debug=False)