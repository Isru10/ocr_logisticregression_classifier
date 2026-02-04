import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

KEYWORDS_ENG = ["license", "tin", "tax", "revenue", "balance", "year"]
KEYWORDS_AMH = ["ፈቃድ", "ታክስ", "ቢዝነስ", "ዓመት"]

def extract_features(text):
    words = text.split()

    try:
        language = detect(text)
    except:
        language = "unknown"

    keyword_hits = 0
    for k in KEYWORDS_ENG + KEYWORDS_AMH:
        if re.search(k, text, re.IGNORECASE):
            keyword_hits += 1

    return {
        "word_count": len(words),
        "char_count": len(text),
        "has_readable_text": 1 if len(words) > 20 else 0,
        "keyword_hits": keyword_hits,
        "language_amh": 1 if language == "am" else 0,
        "language_eng": 1 if language == "en" else 0
    }
