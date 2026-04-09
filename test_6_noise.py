import csv
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from verify import verify
from metrics import compute_FAR, compute_FRR

THRESHOLD = 0.144
N_PER_BAND = 300
SEED = 42
os.makedirs("results", exist_ok=True)

with open("data/test_500.csv") as f:
    all_rows = list(csv.DictReader(f))

# Load baseline
baseline = np.loadtxt("results/baseline_scores.csv", delimiter=",", skiprows=1)
baseline_far = compute_FAR(baseline[:, 0], baseline[:, 1].astype(int), THRESHOLD)
baseline_frr = compute_FRR(baseline[:, 0], baseline[:, 1].astype(int), THRESHOLD)

PSNR_BANDS = [
    ("50-80dB",  50, 80),
    ("40-50dB",  40, 50),
    ("30-40dB",  30, 40),
    ("20-30dB",  20, 30),
    ("10-20dB",  10, 20),
]


def add_gaussian_noise(image, target_psnr_db):
    signal_power = np.mean(image.astype(float) ** 2)
    if signal_power == 0:
        return image
    noise_power = signal_power / (10 ** (target_psnr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)


def run_band(band_name, psnr_low, psnr_high):
    rng = random.Random(SEED)
    np.random.seed(SEED)
    sample = rng.sample(all_rows, min(N_PER_BAND, len(all_rows)))

    scores, labels = [], []
    for row in tqdm(sample, desc=band_name):
        try:
            img = cv2.imread(row["image_path"])
            if img is None:
                raise ValueError("unreadable")
            target_psnr = rng.uniform(psnr_low, psnr_high)
            noisy = add_gaussian_noise(img, target_psnr)
            tmp_path = "results/_tmp_noise.jpg"
            cv2.imwrite(tmp_path, noisy)
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

    out = f"results/noise_{band_name}_scores.csv"
    np.savetxt(out, np.column_stack([scores, labels]),
               delimiter=",", header="score,genuine", comments="")
    return far, frr


print(f"Baseline  FAR={baseline_far:.4f}  FRR={baseline_frr:.4f}  TAR={1-baseline_frr:.4f}")
print()

results = []
for band_name, psnr_low, psnr_high in PSNR_BANDS:
    far, frr = run_band(band_name, psnr_low, psnr_high)
    results.append((band_name, far, frr))

print("\n--- Summary ---")
print(f"{'PSNR band':<12} {'FAR':>8} {'FRR':>8} {'TAR':>8} {'ΔFAR':>8} {'ΔFRR':>8}")
for band_name, far, frr in results:
    print(f"{band_name:<12} {far:>8.4f} {frr:>8.4f} {1-frr:>8.4f} "
          f"{far-baseline_far:>+8.4f} {frr-baseline_frr:>+8.4f}")
