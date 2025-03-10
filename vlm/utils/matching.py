import json
import copy
from thefuzz import fuzz


def find_best_match(llm_text, ocr_texts):
    best_matches = []
    if not llm_text:
        return [], []

    llm_text_lower = llm_text.lower().strip()

    for ocr_text, confidence in ocr_texts:
        ocr_text_lower = ocr_text.lower().strip()
        if len(ocr_text_lower) * 0.3 > len(llm_text_lower):
            continue
        similarity_ratio = fuzz.token_set_ratio(llm_text_lower, ocr_text_lower)

        if similarity_ratio > 80:
            replaced = False
            
            for i, (existing_text, __) in enumerate(best_matches):
                existing_ratio = fuzz.token_set_ratio(existing_text.lower().strip(), ocr_text_lower)
                
                if existing_ratio > 75:  
                    existing_llm_ratio = fuzz.token_set_ratio(llm_text_lower, existing_text.lower().strip())
                    
                    if similarity_ratio > existing_llm_ratio:
                        best_matches[i] = (ocr_text, confidence)
                    replaced = True
                    break

            if not replaced:
                if len(llm_text_lower.split()) == len(best_matches) == 1:
                    break
                best_matches.append((ocr_text, confidence))

    return [confidence for __, confidence in best_matches], [ocr_text for ocr_text, __ in best_matches]


def assign_confidence(llm_result, ocr_result):
    def process_value(llm_value):
        if isinstance(llm_value, dict):
            if "text" in llm_value and "page" in llm_value:
                page_num = llm_value["page"]
                ocr_texts = [(item["text"], item["confidence"]) for item in ocr_result.get(f"page_{page_num}", [])]
                llm_value["confidence"], __ = find_best_match(llm_value["text"], ocr_texts)
                # llm_value["confidence"] = find_best_match(llm_value["text"], ocr_texts)
            else:
                for value in llm_value.values():
                    process_value(value)
        elif isinstance(llm_value, list):
            for item in llm_value:
                process_value(item)

    llm_result_copy = copy.deepcopy(llm_result)
    process_value(llm_result_copy)
    return llm_result_copy


if __name__ == '__main__':
    llm_result = json.load(open("/home/tuongtn/workspace/Projects/Shinhan_OCR/llm.json"))
    ocr_result = json.load(open("/home/tuongtn/workspace/Projects/Shinhan_OCR/ocr.json"))
    json.dump(assign_confidence(llm_result, ocr_result), open("test.json", "w"), ensure_ascii=False)