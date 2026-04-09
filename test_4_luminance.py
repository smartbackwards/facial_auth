import csv
import os
import numpy as np
import cv2
from tqdm import tqdm
from verify import verify
from metrics import compute_FAR, compute_FRR

THRESHOLD = 0.144
os.makedirs("results", exist_ok=True)

with open("data/test_500.csv") as f:
    rows = list(csv.DictReader(f))

# Load baseline for comparison
baseline = np.loadtxt("results/baseline_scores.csv", delimiter=",", skiprows=1)
baseline_scores = baseline[:, 0]
baseline_labels = baseline[:, 1].astype(int)
baseline_far = compute_FAR(baseline_scores, baseline_labels, THRESHOLD)
baseline_frr = compute_FRR(baseline_scores, baseline_labels, THRESHOLD)


def modify_luminance(image_bgr, mode, param=None):
    """Apply luminance modification in YCbCr space."""
    ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(float)
    Y = ycbcr[:, :, 0]
    if mode == "quadratic":
        Y = (Y / 255.0) ** 2 * 255.0
    elif mode == "linear":
        Y = Y * param
    elif mode == "offset":
        Y = Y + param
    ycbcr[:, :, 0] = np.clip(Y, 0, 255)
    return cv2.cvtColor(ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def run_variant(label, modify_fn):
    scores, labels = [], []
    for row in rows:
        try:
            img = cv2.imread(row["image_path"])
            if img is None:
                raise ValueError("unreadable")
            modified = modify_fn(img)
            tmp_path = "results/_tmp_luminance.jpg"
            cv2.imwrite(tmp_path, modified)
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
    print(f"  {label:<30} FAR={far:.4f}  FRR={frr:.4f}  TAR={1-frr:.4f}")
    return far, frr, scores, labels


variants = [
    ("quadratic",         lambda img: modify_luminance(img, "quadratic")),
    ("linear x0.5",       lambda img: modify_luminance(img, "linear", 0.5)),
    ("linear x0.6",       lambda img: modify_luminance(img, "linear", 0.6)),
    ("linear x0.75",      lambda img: modify_luminance(img, "linear", 0.75)),
    ("linear x1.33",      lambda img: modify_luminance(img, "linear", 4/3)),
    ("linear x1.5",       lambda img: modify_luminance(img, "linear", 1.5)),
    ("offset -100",       lambda img: modify_luminance(img, "offset", -100)),
    ("offset -20",        lambda img: modify_luminance(img, "offset", -20)),
    ("offset -10",        lambda img: modify_luminance(img, "offset", -10)),
    ("offset +30",        lambda img: modify_luminance(img, "offset", 30)),
]

print(f"--- Baseline (no modification)        FAR={baseline_far:.4f}  FRR={baseline_frr:.4f}  TAR={1-baseline_frr:.4f} ---")
print()

results = []
for name, fn in variants:
    print(f"Running: {name}")
    far, frr, scores, labels = run_variant(name, fn)
    results.append((name, far, frr))
    np.savetxt(
        f"results/luminance_{name.replace(' ', '_')}_scores.csv",
        np.column_stack([scores, labels]),
        delimiter=",", header="score,genuine", comments="",
    )

print("\n--- Summary ---")
print(f"{'Variant':<30} {'FAR':>8} {'FRR':>8} {'TAR':>8} {'ΔFAR':>8} {'ΔFRR':>8}")
for name, far, frr in results:
    print(f"{name:<30} {far:>8.4f} {frr:>8.4f} {1-frr:>8.4f} "
          f"{far-baseline_far:>+8.4f} {frr-baseline_frr:>+8.4f}")
