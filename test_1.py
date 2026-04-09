import pandas as pd
import numpy as np
from verify import verify
from tqdm import tqdm
df = pd.read_csv("data/test_500.csv")
scores, labels = [], []
for _, row in tqdm(df.iterrows()):
    try:
        _, score = verify(row["image_path"], row["claimed_identity"])
        scores.append(score)
        labels.append(row["attempt_type"]=="genuine")
    except:
        scores.append(0.0)
        labels.append(row["attempt_type"]=="genuine")

scores = np.array(scores)
labels = np.array(labels)
np.savetxt("results/baseline_scores.csv", np.column_stack([scores, labels]),
           delimiter=",", header="score,genuine")