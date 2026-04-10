import csv
import os
import numpy as np
from metrics import compute_FAR, compute_FRR

THRESHOLD = 0.1443
os.makedirs('results', exist_ok=True)

baseline = np.loadtxt('results/baseline_scores.csv', delimiter=',', skiprows=1)
baseline_far = compute_FAR(baseline[:, 0], baseline[:, 1].astype(int), THRESHOLD)
baseline_frr = compute_FRR(baseline[:, 0], baseline[:, 1].astype(int), THRESHOLD)

rows = []
rows.append(['Baseline', baseline_far, baseline_frr, 1 - baseline_frr, 0.0, 0.0])

unknown = np.loadtxt('results/unknown_scores.csv', delimiter=',', skiprows=1)
unknown_far = compute_FAR(unknown[:, 0], unknown[:, 1].astype(int), THRESHOLD)
rows.append(['Unknown users (+100)', unknown_far, '', '', unknown_far - baseline_far, ''])

for label, path in [
    ('Luminance: quadratic', 'results/luminance_quadratic_scores.csv'),
    ('Luminance: linear x0.5', 'results/luminance_linear_x0.5_scores.csv'),
    ('Luminance: linear x0.6', 'results/luminance_linear_x0.6_scores.csv'),
    ('Luminance: linear x0.75', 'results/luminance_linear_x0.75_scores.csv'),
    ('Luminance: linear x1.33', 'results/luminance_linear_x1.33_scores.csv'),
    ('Luminance: linear x1.5', 'results/luminance_linear_x1.5_scores.csv'),
    ('Luminance: offset -100', 'results/luminance_offset_-100_scores.csv'),
    ('Luminance: offset -20', 'results/luminance_offset_-20_scores.csv'),
    ('Luminance: offset -10', 'results/luminance_offset_-10_scores.csv'),
    ('Luminance: offset +30', 'results/luminance_offset_+30_scores.csv'),
    ('Noise: 50-80 dB', 'results/noise_50-80dB_scores.csv'),
    ('Noise: 40-50 dB', 'results/noise_40-50dB_scores.csv'),
    ('Noise: 30-40 dB', 'results/noise_30-40dB_scores.csv'),
    ('Noise: 20-30 dB', 'results/noise_20-30dB_scores.csv'),
    ('Noise: 10-20 dB', 'results/noise_10-20dB_scores.csv'),
    ('JPEG q=95', 'results/jpeg_q95_scores.csv'),
    ('JPEG q=75', 'results/jpeg_q75_scores.csv'),
    ('JPEG q=50', 'results/jpeg_q50_scores.csv'),
    ('JPEG q=35', 'results/jpeg_q35_scores.csv'),
    ('JPEG q=20', 'results/jpeg_q20_scores.csv'),
]:
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    far = compute_FAR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    frr = compute_FRR(data[:, 0], data[:, 1].astype(int), THRESHOLD)
    rows.append([label, far, frr, 1 - frr, far - baseline_far, frr - baseline_frr])

with open('results/results_summary_table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['test_name', 'far', 'frr', 'tar', 'delta_far_vs_baseline', 'delta_frr_vs_baseline'])
    writer.writerows(rows)

if os.path.exists('results/timing_summary.csv'):
    print('Found results/timing_summary.csv')
print('Saved results/results_summary_table.csv')
