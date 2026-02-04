import numpy as np

def build_submission_vector(doc_features):
    return np.array([
        len(doc_features),                              # num documents
        sum(f["word_count"] for f in doc_features),
        sum(f["has_readable_text"] for f in doc_features),
        sum(f["keyword_hits"] for f in doc_features),
        sum(f["language_amh"] for f in doc_features),
        sum(f["language_eng"] for f in doc_features)
    ])
