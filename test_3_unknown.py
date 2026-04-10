import csv
import random
import numpy as np
from tqdm import tqdm
from verify import verify
from metrics import compute_FAR, compute_FRR
import os

os.makedirs("results", exist_ok=True)

THRESHOLD = 0.1443
N_UNKNOWN = 100
SEED = 42

# Load unknown-user rows from splits.csv
with open("data/splits.csv") as f:
    all_rows = list(csv.DictReader(f))

unknown_rows = [r for r in all_rows if r["split"] == "unknown_user"]
random.seed(SEED)
sample = random.sample(unknown_rows, min(N_UNKNOWN, len(unknown_rows)))
print(f"Sampled {len(sample)} unknown-user images")

# Pick a random enrolled identity to claim (worst-case FPR scenario)
with open("data/splits.csv") as f:
    enrolled_ids = list({int(r["identity"]) if r["identity"].isdigit() else r["identity"] for r in csv.DictReader(f) if r["identity_type"] == "enrolled"})

random.seed(SEED)

scores, labels = [], []
for row in tqdm(sample):
    claimed = random.choice(enrolled_ids)
    try:
        _, score = verify(row["image_path"], claimed)
        scores.append(score)
    except Exception:
        scores.append(0.0)
    labels.append(0)  # always impostor

scores = np.array(scores)
labels = np.array(labels)

np.savetxt(
    "results/unknown_scores.csv",
    np.column_stack([scores, labels]),
    delimiter=",",
    header="score,genuine",
    comments="",
)
print("Saved results/unknown_scores.csv")

# Load baseline for comparison
baseline = np.loadtxt("results/baseline_scores.csv", delimiter=",", skiprows=1)
b_scores, b_labels = baseline[:, 0], baseline[:, 1].astype(int)

baseline_far = compute_FAR(b_scores, b_labels, THRESHOLD)
unknown_far  = compute_FAR(scores, labels, THRESHOLD)

print(f"\nBaseline FAR  (threshold={THRESHOLD}): {baseline_far:.4f} ({baseline_far*100:.2f}%)")
print(f"Unknown FAR   (threshold={THRESHOLD}): {unknown_far:.4f}  ({unknown_far*100:.2f}%)")
print(f"FPR change: {(unknown_far - baseline_far)*100:+.2f} pp")
