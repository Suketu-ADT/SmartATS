import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

import os

# Load the model once to avoid reloading on every function call
model_path = os.path.join("models", "fine_tuned_ats_model")
if os.path.exists(model_path):
    print(f"Loading custom fine-tuned ML model from {model_path}...")
    model = SentenceTransformer(model_path)
else:
    print("Loading base ML model (all-MiniLM-L6-v2) for semantic matching...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    """Extracts all text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def preprocess_text(text):
    """Basic text cleaning."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep some basic punctuation that might be structural
    text = re.sub(r'[^a-zA-Z0-9\s.,]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_ats_score(resume_text, job_description_text):
    """
    Calculates the ATS score based on semantic similarity.
    Returns a score from 0 to 100.
    """
    # Preprocess texts
    clean_resume = preprocess_text(resume_text)
    clean_jd = preprocess_text(job_description_text)
    
    if not clean_resume or not clean_jd:
        return 0.0

    # Generate Embeddings
    # Encode both strings to get their vector representations
    embeddings = model.encode([clean_resume, clean_jd])
    
    # Calculate cosine similarity
    resume_embedding = embeddings[0].reshape(1, -1)
    jd_embedding = embeddings[1].reshape(1, -1)
    
    similarity_matrix = cosine_similarity(resume_embedding, jd_embedding)
    
    # Extract the similarity value (0.0 to 1.0)
    similarity_score = similarity_matrix[0][0]
    
    # Convert to percentage (ensure no negative values due to floating point precision)
    percentage_score = max(0.0, similarity_score) * 100
    
    return round(percentage_score, 2)
