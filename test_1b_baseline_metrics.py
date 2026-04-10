import numpy as np
from metrics import compute_FAR, compute_FRR

THRESHOLD = 0.1443

data = np.loadtxt("results/baseline_scores.csv", delimiter=",", skiprows=1)
scores = data[:, 0]
labels = data[:, 1].astype(int)

far = compute_FAR(scores, labels, THRESHOLD)
frr = compute_FRR(scores, labels, THRESHOLD)
tar = 1 - frr

print(f"--- Baseline metrics (threshold={THRESHOLD}) ---")
print(f"FAR : {far:.4f}  ({far*100:.2f}%)")
print(f"FRR : {frr:.4f}  ({frr*100:.2f}%)")
print(f"TAR : {tar:.4f}  ({tar*100:.2f}%)")
