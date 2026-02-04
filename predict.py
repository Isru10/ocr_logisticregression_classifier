import os
import joblib
from ocr import run_ocr
from sentence_transformers import SentenceTransformer

MODEL_PATH = "spam_classifier_v1.joblib"
SUBMISSION_DIR = "submission"

clf = joblib.load(MODEL_PATH)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🔮 Prediction started...\n")

for submission_id in os.listdir(SUBMISSION_DIR):
    sub_path = os.path.join(SUBMISSION_DIR, submission_id)
    if not os.path.isdir(sub_path):
        continue

    combined_text = ""
    for file in os.listdir(sub_path):
        if file.endswith(".png"):
            combined_text += " " + run_ocr(os.path.join(sub_path, file))

    vector = embedder.encode(combined_text)
    probs = clf.predict_proba([vector])[0]
    pred = clf.predict([vector])[0]

    print(f"{submission_id}")
    print(f"  Spam probability: {probs[0]:.3f}")
    print(f"  Not spam probability: {probs[1]:.3f}")
    print(f"  Prediction: {'NOT SPAM' if pred == 1 else 'SPAM'}\n")



# import joblib
# from ocr import run_ocr
# from features import extract_features
# from vectorize import build_submission_vector
# import os

# MODEL_PATH = "model.joblib"
# SUBMISSION_DIR = "submission"

# def classify_submission():
#     model = joblib.load(MODEL_PATH)

#     doc_features = []

#     # 👇 THIS is where tin.png, trade_license.png, etc. are used
#     for file_name in os.listdir(SUBMISSION_DIR):
#         if file_name.endswith(".png"):
#             image_path = os.path.join(SUBMISSION_DIR, file_name)

#             text = run_ocr(image_path)                 # OCR on tin.png etc.
#             features = extract_features(text)           # Extract signals
#             doc_features.append(features)

#     vector = build_submission_vector(doc_features)
#     probs = model.predict_proba([vector])[0]

#     return {
#         "low_quality_score": round(probs[0], 3),
#         "acceptable_score": round(probs[1], 3)
#     }


# if __name__ == "__main__":
#     result = classify_submission()
#     print(result)
