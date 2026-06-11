import os
import re
import pandas as pd
from datasets import load_dataset

def download_resume_dataset(output_filepath):
    """Downloads the resume-jd-match dataset and extracts the components."""
    print("Downloading dataset from Hugging Face...")
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Load the dataset
        dataset = load_dataset("facehuggerapoorv/resume-jd-match")
        
        # We will combine train and test for a single comprehensive dataset
        df_train = pd.DataFrame(dataset['train'])
        df_test = pd.DataFrame(dataset['test'])
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        print(f"Loaded {len(df)} records. Parsing text...")
        
        parsed_data = []
        # The text format is:
        # "For the given job description <<...>> the resume: <<...>>. The result is, (Fit|No Fit)"
        
        pattern = r"For the given job description <<(.*?)>> the resume: <<(.*?)>>\. The result is, (.*)"
        
        for idx, row in df.iterrows():
            text = row['text']
            match = re.search(pattern, text, re.DOTALL)
            if match:
                job_description = match.group(1).strip()
                resume = match.group(2).strip()
                label = match.group(3).strip()
                parsed_data.append({
                    'job_description': job_description,
                    'resume': resume,
                    'label': label
                })
        
        parsed_df = pd.DataFrame(parsed_data)
        
        # Save to CSV
        parsed_df.to_csv(output_filepath, index=False, encoding='utf-8')
        
        print(f"Data saved successfully to {output_filepath}")
        print(f"Total parsed records: {len(parsed_df)}")
        
    except Exception as e:
        print(f"Failed to download data: {e}")

if __name__ == '__main__':
    raw_data_path = os.path.join("data", "raw", "resume_dataset.csv")
    download_resume_dataset(raw_data_path)
