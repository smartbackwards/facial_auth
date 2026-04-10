import csv
import os
import cv2
import numpy as np

THRESHOLD = 0.1443
TOP_K = 5
OUT_DIR = 'results/error_cases'
os.makedirs(OUT_DIR, exist_ok=True)

with open('data/test_500.csv') as f:
    test_rows = list(csv.DictReader(f))

baseline = np.loadtxt('results/baseline_scores.csv', delimiter=',', skiprows=1)
scores = baseline[:, 0]
labels = baseline[:, 1].astype(int)

if len(test_rows) != len(scores):
    raise ValueError('test_500.csv and baseline_scores.csv have different lengths')

with open('data/splits.csv') as f:
    split_rows = list(csv.DictReader(f))

enrollment_rows = {}
for row in split_rows:
    if row['split'] == 'enrollment':
        enrollment_rows[str(row['identity'])] = row

all_cases = []
false_accepts = []
false_rejects = []

for i, row in enumerate(test_rows):
    score = float(scores[i])
    label = int(labels[i])
    predicted_match = score >= THRESHOLD

    true_identity = str(row['identity'])
    claimed_identity = str(row['claimed_identity'])
    enrollment_row = enrollment_rows.get(claimed_identity)
    enrollment_path = enrollment_row['image_path'] if enrollment_row else ''

    case = {
        'index': i,
        'image_name': row['image_name'],
        'image_path': row['image_path'],
        'true_identity': true_identity,
        'claimed_identity': claimed_identity,
        'attempt_type': row['attempt_type'],
        'score': score,
        'threshold': THRESHOLD,
        'predicted_match': int(predicted_match),
        'enrollment_image_path': enrollment_path,
    }
    all_cases.append(case)

    if label == 0 and predicted_match:
        false_accepts.append(case)
    if label == 1 and not predicted_match:
        false_rejects.append(case)

false_accepts.sort(key=lambda x: x['score'], reverse=True)
false_rejects.sort(key=lambda x: x['score'])

top_fa = false_accepts[:TOP_K]
top_fr = false_rejects[:TOP_K]


def save_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'index', 'image_name', 'image_path', 'true_identity', 'claimed_identity',
            'attempt_type', 'score', 'threshold', 'predicted_match', 'enrollment_image_path'
        ])
        writer.writeheader()
        writer.writerows(rows)


def make_panel(case, error_type, rank):
    left = cv2.imread(case['image_path'])
    right = cv2.imread(case['enrollment_image_path']) if case['enrollment_image_path'] else None
    if left is None or right is None:
        return

    h = max(left.shape[0], right.shape[0])
    left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    gap = np.full((h, 20, 3), 255, dtype=np.uint8)
    panel = np.hstack([left, gap, right])
    footer = np.full((120, panel.shape[1], 3), 255, dtype=np.uint8)
    panel = np.vstack([panel, footer])

    cv2.putText(panel, f'{error_type} #{rank}', (20, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(panel, f'score={case["score"]:.4f}  threshold={case["threshold"]:.4f}', (20, h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(panel, f'true={case["true_identity"]}  claimed={case["claimed_identity"]}', (20, h + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(panel, 'left: query image', (20, h + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(panel, 'right: enrollment image of claimed identity', (260, h + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)

    out = os.path.join(OUT_DIR, f'{error_type.lower().replace(" ", "_")}_{rank}.jpg')
    cv2.imwrite(out, panel)


save_csv('results/error_cases_false_accepts.csv', false_accepts)
save_csv('results/error_cases_false_rejects.csv', false_rejects)
save_csv('results/error_cases_false_accepts_top.csv', top_fa)
save_csv('results/error_cases_false_rejects_top.csv', top_fr)

with open('results/error_cases_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['threshold', THRESHOLD])
    writer.writerow(['false_accept_count', len(false_accepts)])
    writer.writerow(['false_reject_count', len(false_rejects)])
    writer.writerow(['top_k_saved', TOP_K])

for i, case in enumerate(top_fa, start=1):
    make_panel(case, 'False Accept', i)
for i, case in enumerate(top_fr, start=1):
    make_panel(case, 'False Reject', i)

print('Saved results/error_cases_false_accepts.csv')
print('Saved results/error_cases_false_rejects.csv')
print('Saved results/error_cases_false_accepts_top.csv')
print('Saved results/error_cases_false_rejects_top.csv')
print('Saved results/error_cases_summary.csv')
print(f'Saved image panels to {OUT_DIR}')
print(f'False accepts: {len(false_accepts)}')
print(f'False rejects: {len(false_rejects)}')
