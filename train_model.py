import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def train():
    print("Loading dataset...")
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
    
    # Use a smaller sample so it runs quickly on a local CPU
    # You can increase this to len(df) if you want to train on everything
    SAMPLE_SIZE = min(1000, len(df))
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Prepare data for sentence-transformers
    print(f"Preparing {len(train_df)} training examples...")
    train_examples = []
    for _, row in train_df.iterrows():
        train_examples.append(InputExample(texts=[row['job_description'], row['resume']], label=float(row['score'])))
        
    # We use a PyTorch DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    
    # Load base model
    print("Loading base model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Use CosineSimilarityLoss for training similarity mappings
    train_loss = losses.CosineSimilarityLoss(model)
    
    print("Starting training...")
    # Tune epochs and warmup_steps based on your hardware capabilities
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        output_path=os.path.join("models", "fine_tuned_ats_model")
    )
    print("Training complete! Model saved to 'models/fine_tuned_ats_model'.")

if __name__ == "__main__":
    train()
