import time
from flask_cors import CORS
from flask import Flask, request

from vlm.doc_vqa import DocExtractor
from tools import config as cfg


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

doc_extractor = DocExtractor()


@app.route('/api/v1/doc-vqa', methods=['POST'])
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
        st = time.time()
        result = doc_extractor.extract_wt_outformat(pdf=pdf_file,
                                       output_format=output_format,
                                        pdf_type='from_bytes')
        return {
            'status': 'success',
            'message': result,
            'time': time.time() - st
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
    

@app.route('/api/v2/doc-vqa', methods=['POST'])
def doc_vqa_v2():
    if 'pdf' not in request.files:
        return {
            'status': 'error',
            'message': 'Not found pdf file'
        }
    
    pdf_file = request.files['pdf']
    prompt = request.form.get('prompt', '')

    if len(prompt) == 0:
        return {
            'status': 'error',
            'message': 'Not exist prompt',
        }
    
    try:
        st = time.time()
        result = doc_extractor.extract_wt_prompt(pdf=pdf_file,
                                        prompt=prompt,
                                        pdf_type='from_bytes')
        return {
            'status': 'success',
            'message': result,
            'time': time.time() - st
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
        

@app.route('/api/v3/doc-vqa', methods=['POST'])
def doc_vqa_v3():
    if 'pdf' not in request.files:
        return {
            'status': 'error',
            'message': 'Not found pdf file'
        }
    
    pdf_file = request.files['pdf']
    output_format = request.form.get('output_format', None)
    ocr_token = request.form.get('ocr_token', [])
    
    if output_format is None:
        return {
            'status': 'error',
            'message': 'Not found output format',
        }
    try:
        st = time.time()
        result = doc_extractor.extract_wt_ocrtoken(pdf=pdf_file,
                                       output_format=output_format,
                                       ocr_token=ocr_token,
                                        pdf_type=None)
        return {
            'status': 'success',
            'message': result,
            'time': time.time() - st
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=cfg['Deploy']['port'], debug=False)