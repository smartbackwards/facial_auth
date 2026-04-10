import csv
import os
import pickle

os.makedirs('results', exist_ok=True)

with open('data/splits.csv') as f:
    split_rows = list(csv.DictReader(f))

with open('data/test_500.csv') as f:
    test_rows = list(csv.DictReader(f))

with open('celeba_subset/enrolled_identities.csv') as f:
    enrolled_id_rows = list(csv.DictReader(f))

with open('celeba_subset/unknown_identities.csv') as f:
    unknown_id_rows = list(csv.DictReader(f))

with open('database/embeddings.pkl', 'rb') as f:
    db = pickle.load(f)

split_counts = {}
for row in split_rows:
    key = row['split']
    split_counts[key] = split_counts.get(key, 0) + 1

identity_type_counts = {}
for row in split_rows:
    key = row['identity_type']
    identity_type_counts[key] = identity_type_counts.get(key, 0) + 1

split_identity_sets = {}
for row in split_rows:
    key = row['split']
    split_identity_sets.setdefault(key, set()).add(row['identity'])

attempt_counts = {}
for row in test_rows:
    key = row['attempt_type']
    attempt_counts[key] = attempt_counts.get(key, 0) + 1

manual_db_keys = [str(k) for k in db.keys() if not str(k).isdigit()]
manual_key_count = len(manual_db_keys)

summary_rows = [
    ('enrolled_identities_declared', len(enrolled_id_rows)),
    ('unknown_identities_declared', len(unknown_id_rows)),
    ('db_profiles_total', len(db)),
    ('db_profiles_manual_named', manual_key_count),
    ('split_enrollment_images', split_counts.get('enrollment', 0)),
    ('split_genuine_test_images', split_counts.get('genuine_test', 0)),
    ('split_impostor_pool_images', split_counts.get('impostor_pool', 0)),
    ('split_unknown_user_images', split_counts.get('unknown_user', 0)),
    ('split_enrollment_identities', len(split_identity_sets.get('enrollment', set()))),
    ('split_genuine_test_identities', len(split_identity_sets.get('genuine_test', set()))),
    ('split_impostor_pool_identities', len(split_identity_sets.get('impostor_pool', set()))),
    ('split_unknown_user_identities', len(split_identity_sets.get('unknown_user', set()))),
    ('test500_total_attempts', len(test_rows)),
    ('test500_genuine_attempts', attempt_counts.get('genuine', 0)),
    ('test500_impostor_attempts', attempt_counts.get('impostor', 0)),
]

with open('results/dataset_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerows(summary_rows)

with open('results/dataset_split_table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split', 'image_count', 'identity_count'])
    for split_name in ['enrollment', 'genuine_test', 'impostor_pool', 'unknown_user']:
        writer.writerow([
            split_name,
            split_counts.get(split_name, 0),
            len(split_identity_sets.get(split_name, set())),
        ])

with open('results/manual_db_profiles.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['db_key'])
    for key in manual_db_keys:
        writer.writerow([key])

print('Saved results/dataset_summary.csv')
print('Saved results/dataset_split_table.csv')
print('Saved results/manual_db_profiles.csv')
print('\n--- Dataset summary ---')
for metric, value in summary_rows:
    print(f'{metric:<32} {value}')
