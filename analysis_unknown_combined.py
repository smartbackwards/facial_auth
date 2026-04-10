import csv
import numpy as np
from metrics import compute_FAR

THRESHOLD = 0.1443

baseline = np.loadtxt('results/baseline_scores.csv', delimiter=',', skiprows=1)
unknown = np.loadtxt('results/unknown_scores.csv', delimiter=',', skiprows=1)

baseline_scores = baseline[:, 0]
baseline_labels = baseline[:, 1].astype(int)
unknown_scores = unknown[:, 0]
unknown_labels = unknown[:, 1].astype(int)

combined_scores = np.concatenate([baseline_scores, unknown_scores])
combined_labels = np.concatenate([baseline_labels, unknown_labels])

baseline_far = compute_FAR(baseline_scores, baseline_labels, THRESHOLD)
unknown_far = compute_FAR(unknown_scores, unknown_labels, THRESHOLD)
combined_far = compute_FAR(combined_scores, combined_labels, THRESHOLD)

baseline_impostors = int(np.sum(baseline_labels == 0))
unknown_impostors = int(np.sum(unknown_labels == 0))
combined_impostors = int(np.sum(combined_labels == 0))

with open('results/unknown_combined_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['set_name', 'threshold', 'sample_count', 'impostor_count', 'far'])
    writer.writerow(['baseline_500', THRESHOLD, len(baseline_scores), baseline_impostors, baseline_far])
    writer.writerow(['unknown_only', THRESHOLD, len(unknown_scores), unknown_impostors, unknown_far])
    writer.writerow(['baseline_plus_unknown', THRESHOLD, len(combined_scores), combined_impostors, combined_far])

print('Saved results/unknown_combined_metrics.csv')
print(f'Baseline FAR:          {baseline_far:.4f}')
print(f'Unknown-only FAR:      {unknown_far:.4f}')
print(f'Baseline+unknown FAR:  {combined_far:.4f}')
