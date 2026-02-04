# 📌 Spam Classification for Pitch Submissions (GC Project)

This repository contains a **machine learning pipeline** used in our **Smart Pitching & Investor Matching platform**.

The goal of this module is to **classify pitch submissions as spam or valid** based on uploaded business documents.

Each pitch submission consists of **three document images (PNG)**:
- **TIN document**
- **Trade License**
- **Financial Statement**

---

## 🧠 What This System Does

1. Performs **OCR** on uploaded document images
2. Extracts readable text
3. Converts text into **dense embeddings** using a pretrained NLP model
4. Trains a **binary classification model**
5. Saves the trained model for later inference

---

## 🧩 Model Architecture

- **OCR Engine**: Tesseract OCR
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **ML Library**: Scikit-learn
- **Classifier**: Logistic Regression
- **Model Output**: `.joblib` file

---

## 📁 Project Structure

.
├── train.py # Training pipeline
├── predict.py # Prediction script
├── labels.csv # Training labels
├── requirements.txt # Python dependencies
├── models/
│ └── spam_classifier_v1.joblib
└── submission/



---

## 🏷️ Dataset Format

### `labels.csv`

```csv
submission_id,doc_files,label
pitch_001,tin.png|trade_license.png|financial_statement.png,1
pitch_002,tin2.png|trade_license2.png|financial_statement2.png,0
pitch_003,tin3.png|trade_license3.png|financial_statement3.png,1
Column Explanation
Column	Description
submission_id	Unique pitch identifier
doc_files	Three document images separated by `
label	1 = valid pitch, 0 = spam
📌 All three documents are treated as ONE submission vector.

🚀 How to Run This Project
1️⃣ Clone the Repository
git clone https://github.com/Isru10/ocr_logisticregression_classifier.git
cd ocr_logisticregression_classifier
2️⃣ Create a Virtual Environment
python -m venv .venv
3️⃣ Activate the Environment
Windows
.venv\Scripts\activate


Mac / Linux
source .venv/bin/activate
4️⃣ Install Dependencies
pip install -r requirements.txt
5️⃣ Train the Model
python train.py or uv run python train.py
📊 Training Output
Console will display precision, recall, F1-score, accuracy

Trained model is saved as:

models/spam_classifier_v1.joblib
⚠️ Note: With small datasets, metrics may be unstable.
This is expected and will improve with more data.

🔁 Model Versioning
Each training run produces a new model version

Example:

spam_classifier_v1.joblib
spam_classifier_v2.joblib
Models are not created per image, only per training run.

🧪 Prediction (Optional)
Once trained, you can run:

python predict.py
This loads the saved model and classifies a new submission.

🔮 Planned Next Steps
Increase labeled dataset size

Add FastAPI inference endpoint

Integrate with Node.js backend

Connect to Next.js & Mobile app

Add investor recommendation (RAG module)

📌 Notes
Training is done locally

Model files can be shared across machines

This module focuses only on spam classification

Recommendation logic is handled separately

