import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from metrics import compute_FAR, compute_FRR

THRESHOLD = 0.1443
os.makedirs('results', exist_ok=True)

baseline = np.loadtxt('results/baseline_scores.csv', delimiter=',', skiprows=1)
baseline_scores = baseline[:, 0]
baseline_labels = baseline[:, 1].astype(int)
baseline_far = compute_FAR(baseline_scores, baseline_labels, THRESHOLD)
baseline_frr = compute_FRR(baseline_scores, baseline_labels, THRESHOLD)

# Baseline score histogram
plt.figure(figsize=(8, 5))
plt.hist(baseline_scores[baseline_labels == 1], bins=30, alpha=0.6, label='Genuine')
plt.hist(baseline_scores[baseline_labels == 0], bins=30, alpha=0.6, label='Impostor')
plt.axvline(THRESHOLD, linestyle='--', label=f'Threshold = {THRESHOLD:.4f}')
plt.xlabel('Similarity score')
plt.ylabel('Count')
plt.title('Baseline score distribution')
plt.legend()
plt.tight_layout()
plt.savefig('results/plot_baseline_score_hist.png', dpi=150)
plt.close()

# Noise plot
noise_rows = []
for label, path in [
    ('50-80', 'results/noise_50-80dB_scores.csv'),
    ('40-50', 'results/noise_40-50dB_scores.csv'),
    ('30-40', 'results/noise_30-40dB_scores.csv'),
    ('20-30', 'results/noise_20-30dB_scores.csv'),
    ('10-20', 'results/noise_10-20dB_scores.csv'),
]:
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    far = compute_FAR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    frr = compute_FRR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    noise_rows.append((label, far, frr))

x = np.arange(len(noise_rows))
width = 0.38
plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, [r[1] for r in noise_rows], width=width, label='FAR')
plt.bar(x + width / 2, [r[2] for r in noise_rows], width=width, label='FRR')
plt.axhline(baseline_far, linestyle='--', linewidth=1)
plt.axhline(baseline_frr, linestyle=':', linewidth=1)
plt.xticks(x, [r[0] for r in noise_rows])
plt.xlabel('PSNR band [dB]')
plt.ylabel('Rate')
plt.title('Noise sensitivity')
plt.legend()
plt.tight_layout()
plt.savefig('results/plot_noise_results.png', dpi=150)
plt.close()

# JPEG plot
jpeg_rows = []
for q in [95, 75, 50, 35, 20]:
    data = np.loadtxt(f'results/jpeg_q{q}_scores.csv', delimiter=',', skiprows=1)
    far = compute_FAR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    frr = compute_FRR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    jpeg_rows.append((q, far, frr))

plt.figure(figsize=(8, 5))
plt.plot([r[0] for r in jpeg_rows], [r[1] for r in jpeg_rows], marker='o', label='FAR')
plt.plot([r[0] for r in jpeg_rows], [r[2] for r in jpeg_rows], marker='o', label='FRR')
plt.axhline(baseline_far, linestyle='--', linewidth=1)
plt.axhline(baseline_frr, linestyle=':', linewidth=1)
plt.xlabel('JPEG quality')
plt.ylabel('Rate')
plt.title('JPEG compression sensitivity')
plt.legend()
plt.tight_layout()
plt.savefig('results/plot_jpeg_results.png', dpi=150)
plt.close()

# Luminance plot
lum_paths = [
    ('quadratic', 'results/luminance_quadratic_scores.csv'),
    ('x0.5', 'results/luminance_linear_x0.5_scores.csv'),
    ('x0.6', 'results/luminance_linear_x0.6_scores.csv'),
    ('x0.75', 'results/luminance_linear_x0.75_scores.csv'),
    ('x1.33', 'results/luminance_linear_x1.33_scores.csv'),
    ('x1.5', 'results/luminance_linear_x1.5_scores.csv'),
    ('-100', 'results/luminance_offset_-100_scores.csv'),
    ('-20', 'results/luminance_offset_-20_scores.csv'),
    ('-10', 'results/luminance_offset_-10_scores.csv'),
    ('+30', 'results/luminance_offset_+30_scores.csv'),
]
lum_rows = []
for label, path in lum_paths:
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    far = compute_FAR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    frr = compute_FRR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    lum_rows.append((label, far, frr))

x = np.arange(len(lum_rows))
width = 0.38
plt.figure(figsize=(11, 5))
plt.bar(x - width / 2, [r[1] for r in lum_rows], width=width, label='FAR')
plt.bar(x + width / 2, [r[2] for r in lum_rows], width=width, label='FRR')
plt.axhline(baseline_far, linestyle='--', linewidth=1)
plt.axhline(baseline_frr, linestyle=':', linewidth=1)
plt.xticks(x, [r[0] for r in lum_rows])
plt.xlabel('Luminance variant')
plt.ylabel('Rate')
plt.title('Luminance sensitivity')
plt.legend()
plt.tight_layout()
plt.savefig('results/plot_luminance_results.png', dpi=150)
plt.close()

# Timing plot
with open('results/timing_summary.csv') as f:
    timing_rows = list(csv.DictReader(f))
ops = [r['operation'] for r in timing_rows]
medians = [float(r['median_s']) for r in timing_rows]
means = [float(r['mean_s']) for r in timing_rows]

x = np.arange(len(ops))
width = 0.38
plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, means, width=width, label='Mean')
plt.bar(x + width / 2, medians, width=width, label='Median')
plt.xticks(x, ops)
plt.ylabel('Time [s]')
plt.title('Timing summary')
plt.legend()
plt.tight_layout()
plt.savefig('results/plot_timing_results.png', dpi=150)
plt.close()

print('Saved results/plot_baseline_score_hist.png')
print('Saved results/plot_noise_results.png')
print('Saved results/plot_jpeg_results.png')
print('Saved results/plot_luminance_results.png')
print('Saved results/plot_timing_results.png')
