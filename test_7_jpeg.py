import csv
import os
import numpy as np
import cv2
from tqdm import tqdm
from verify import verify
from metrics import compute_FAR, compute_FRR

THRESHOLD = 0.144
JPEG_QUALITIES = [95, 75, 50, 35, 20]
os.makedirs("results", exist_ok=True)

with open("data/test_500.csv") as f:
    rows = list(csv.DictReader(f))

baseline = np.loadtxt("results/baseline_scores.csv", delimiter=",", skiprows=1)
baseline_scores = baseline[:, 0]
baseline_labels = baseline[:, 1].astype(int)
baseline_far = compute_FAR(baseline_scores, baseline_labels, THRESHOLD)
baseline_frr = compute_FRR(baseline_scores, baseline_labels, THRESHOLD)


def run_quality(quality):
    scores, labels = [], []
    for row in tqdm(rows, desc=f"JPEG q={quality}"):
        try:
            img = cv2.imread(row["image_path"])
            if img is None:
                raise ValueError("unreadable")
            tmp_path = "results/_tmp_jpeg.jpg"
            cv2.imwrite(tmp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            claimed = int(row["claimed_identity"]) if row["claimed_identity"].isdigit() else row["claimed_identity"]
            _, score = verify(tmp_path, claimed)
            scores.append(score)
        except Exception:
            scores.append(0.0)
        labels.append(1 if row["attempt_type"] == "genuine" else 0)

    scores = np.array(scores)
    labels = np.array(labels)
    far = compute_FAR(scores, labels, THRESHOLD)
    frr = compute_FRR(scores, labels, THRESHOLD)

    out = f"results/jpeg_q{quality}_scores.csv"
    np.savetxt(out, np.column_stack([scores, labels]),
               delimiter=",", header="score,genuine", comments="")
    return far, frr


print(f"Baseline  FAR={baseline_far:.4f}  FRR={baseline_frr:.4f}  TAR={1-baseline_frr:.4f}")
print()

results = []
for quality in JPEG_QUALITIES:
    far, frr = run_quality(quality)
    results.append((quality, far, frr))

print("\n--- Summary ---")
print(f"{'JPEG quality':<14} {'FAR':>8} {'FRR':>8} {'TAR':>8} {'ΔFAR':>8} {'ΔFRR':>8}")
for quality, far, frr in results:
    print(f"{quality:<14} {far:>8.4f} {frr:>8.4f} {1-frr:>8.4f} "
          f"{far-baseline_far:>+8.4f} {frr-baseline_frr:>+8.4f}")
