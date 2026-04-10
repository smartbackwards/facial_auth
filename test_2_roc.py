import numpy as np
import matplotlib.pyplot as plt
from metrics import compute_FAR, compute_FRR, threshold_sweep
import os

os.makedirs("results", exist_ok=True)

data = np.loadtxt("results/baseline_scores.csv", delimiter=",", skiprows=1)
scores = data[:, 0]
labels = data[:, 1].astype(int)

thresholds, fars, frrs = threshold_sweep(scores, labels, steps=500)

# Find EER (where FAR ≈ FRR)
eer_idx = np.argmin(np.abs(fars - frrs))
eer_threshold = thresholds[eer_idx]
eer = (fars[eer_idx] + frrs[eer_idx]) / 2

print(f"EER:       {eer:.4f} ({eer*100:.2f}%)")
print(f"Threshold: {eer_threshold:.4f}")
print(f"  FAR at EER: {fars[eer_idx]:.4f}")
print(f"  FRR at EER: {frrs[eer_idx]:.4f}")

# FAR/FRR vs threshold
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, fars, label="FAR")
ax.plot(thresholds, frrs, label="FRR")
ax.axvline(eer_threshold, color="gray", linestyle="--", label=f"EER threshold = {eer_threshold:.3f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Rate")
ax.set_title("FAR / FRR vs Decision Threshold")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("results/far_frr_vs_threshold.png", dpi=150)
print("Saved results/far_frr_vs_threshold.png")

# ROC curve (TAR vs FAR)
tars = 1 - frrs
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fars, tars, label="ROC")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
ax.scatter([fars[eer_idx]], [tars[eer_idx]], color="red", zorder=5, label=f"EER ({eer*100:.1f}%)")
ax.set_xlabel("FAR (False Accept Rate)")
ax.set_ylabel("TAR (True Accept Rate)")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("results/roc_curve.png", dpi=150)
print("Saved results/roc_curve.png")
