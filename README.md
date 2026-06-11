# SmartATS 🎯

SmartATS is a Machine Learning-powered Applicant Tracking System (ATS) Score Detector built using Python, Streamlit, and Hugging Face's `sentence-transformers`. It calculates the semantic similarity between a Job Description and a Candidate's Resume to predict how well the candidate fits the role.

## Features
- **Semantic Similarity Scoring**: Utilizes the `all-MiniLM-L6-v2` Sentence Transformer model to encode text and calculate Cosine Similarity between the Job Description and Resume.
- **Custom ML Pipeline**: Includes end-to-end scripts to download a real-world dataset, fine-tune the base model using Contrastive Learning, and evaluate the custom model's performance.
- **Interactive UI**: A simple, user-friendly Streamlit web application where users can paste job descriptions, upload PDF resumes, and receive instant feedback.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Suketu-ADT/SmartATS.git
cd "SmartATS"
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Launching the Web App
To run the ATS Score Detector interface locally:
```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Pipeline

If you'd like to fine-tune your own custom scoring model, you can use the included pipeline scripts.

### 1. Download Dataset
This will securely download an 8,000-record dataset (`facehuggerapoorv/resume-jd-match`) from Hugging Face and prepare it for training.
```bash
python make_dataset.py
```

### 2. Train Model
This script maps string labels (Fit / No Fit) to semantic scores and fine-tunes the base model utilizing a `CosineSimilarityLoss` function. The trained model is automatically saved to the `models/fine_tuned_ats_model` directory.
```bash
python train_model.py
```

### 3. Evaluate Model
Measure your custom model's accuracy against a hold-out test set using Pearson/Spearman correlation metrics.
```bash
python test_model.py
```

*Note: Once you successfully train a model, `app.py` will automatically detect the custom model in the `models/` directory and use it to score resumes instead of the base model!*
