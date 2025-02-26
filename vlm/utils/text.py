from thefuzz import fuzz
import json
import re


def similarity_text(text1, text2):
    res = fuzz.ratio(str(text1), str(text2))
    return res


def remove_whitespace(text):
    text = re.sub("[\n]+", "\n", text)
    text = re.sub("[\s]+", " ", text)
    return text


def make_json(text):
    if not text:
        return None
    r = None
    for end in ['', '"}', '"]}']:
        try:
            t = text.strip() + end
            d = re.search('\{.*\}', remove_whitespace(t)).group()
            r = json.loads(d, strict=False)
            return r
        except:
            pass
