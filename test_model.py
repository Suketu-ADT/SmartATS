import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split

def test():
    print("Loading dataset for evaluation...")
    df = pd.read_csv(os.path.join("data", "raw", "resume_dataset.csv"))
    
    # Drop rows with missing labels
    df = df.dropna(subset=['label'])
    
    # Map labels to float scores
    label_map = {
        'No Fit': 0.0,
        'Potential Fit': 0.5,
        'Good Fit': 1.0
    }
    df['score'] = df['label'].map(label_map)
    df = df.dropna(subset=['score'])
    
    # Use the same sample size and random state to ensure we get the same split
    SAMPLE_SIZE = min(1000, len(df))
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # Split into train and validation (test)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    model_path = os.path.join("models", "fine_tuned_ats_model")
    if os.path.exists(model_path):
        print(f"Loading custom fine-tuned model from {model_path}...")
        model = SentenceTransformer(model_path)
    else:
        print("Custom model not found. Loading base model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
    print(f"Evaluating on {len(test_df)} samples...")
    sentences1 = test_df['job_description'].tolist()
    sentences2 = test_df['resume'].tolist()
    scores = test_df['score'].tolist()
    
    # Compute the correlation between the model's similarity score and the actual label score
    evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    
    # Run evaluation
    results = evaluator(model)
    
    print("\n--- Evaluation Results ---")
    if isinstance(results, dict):
        for metric, score in results.items():
            print(f"{metric.capitalize()}: {score:.4f}")
    else:
        print(f"Correlation Score (higher is better): {results:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    test()
