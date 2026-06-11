import pandas as pd
from datasets import load_dataset

try:
    print("Trying facehuggerapoorv/resume-jd-match:")
    ds = load_dataset("facehuggerapoorv/resume-jd-match")
    print(ds)
    print(ds['train'][0])
except Exception as e:
    print(e)

print("\n----------------\n")

try:
    print("Trying netsol/resume-score-details:")
    ds2 = load_dataset("netsol/resume-score-details")
    print(ds2)
    print(ds2['train'][0])
except Exception as e:
    print(e)
