import csv
import os
import joblib
import numpy as np
from ocr import run_ocr
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

SUBMISSION_DIR = "submission"
LABELS_FILE = "labels.csv"
MODEL_PATH = "spam_classifier_v1.joblib"

# Load labels
submissions = []

with open(LABELS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        submissions.append({
            "submission_id": row["submission_id"],
            "doc_files": row["doc_files"].split("|"),
            "label": int(row["label"])
        })

# Load embedder
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

X = []
y = []

print("🔍 OCR + Embedding started...")

for sub in submissions:
    combined_text = ""
    base_path = os.path.join(SUBMISSION_DIR, sub["submission_id"])

    for doc in sub["doc_files"]:
        img_path = os.path.join(base_path, doc)
        combined_text += " " + run_ocr(img_path)

    vector = embedder.encode(combined_text)
    X.append(vector)
    y.append(sub["label"])

X = np.array(X)
y = np.array(y)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Save model
joblib.dump(clf, MODEL_PATH)

# Evaluate on training data (first run only)
preds = clf.predict(X)
print("\n📊 Training Results (first run):")
print(classification_report(y, preds))

print(f"\n✅ Model saved as {MODEL_PATH}")


# manual training

# from sklearn.linear_model import LogisticRegression
# import joblib

# # Feature order:
# # [num_docs, total_words, readable_docs, keyword_hits, amh_docs, eng_docs]

# X = [
#     [3, 1200, 3, 5, 1, 2],   # good (mixed language)
#     [3, 80,   1, 0, 1, 0],   # bad (almost empty)
#     [2, 300,  2, 1, 2, 0],   # missing docs
#     [3, 900,  3, 4, 0, 3]    # good (English)
# ]

# y = [1, 0, 0, 1]  # 1 = acceptable, 0 = low quality

# model = LogisticRegression()
# model.fit(X, y)

# joblib.dump(model, "model.joblib")
# print("Model trained and saved")


# feature_names = [
#     "num_documents",
#     "total_word_count",
#     "documents_with_text",
#     "keyword_hits",
#     "amharic_docs",
#     "english_docs"
# ]

# for name, coef in zip(feature_names, model.coef_[0]):
#     print(name, coef)
