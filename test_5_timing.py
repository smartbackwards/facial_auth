import csv
import os
import time
import pickle
import shutil
import numpy as np
import enroll as enroll_module
from enroll import enroll, get_embedding
from verify import verify, cosine_similarity

THRESHOLD = 0.144
N_ENROLL = 20
N_VERIFY = 100
N_IDENTIFY = 100
TEMP_DB_PATH = "results/_tmp_timing_embeddings.pkl"
OUT_CSV = "results/timing_samples.csv"
SUMMARY_CSV = "results/timing_summary.csv"

os.makedirs("results", exist_ok=True)

with open("data/splits.csv") as f:
    split_rows = list(csv.DictReader(f))

with open("data/test_500.csv") as f:
    test_rows = list(csv.DictReader(f))


def load_temp_db():
    with open(TEMP_DB_PATH, "rb") as f:
        return pickle.load(f)


def identify(image_path):
    db = load_temp_db()
    query_emb = get_embedding(image_path)
    best_id = None
    best_score = -1.0
    for user_id, enrolled_emb in db.items():
        score = cosine_similarity(query_emb, enrolled_emb)
        if score > best_score:
            best_score = score
            best_id = user_id
    return best_id, float(best_score)


def summarize(times):
    arr = np.array(times, dtype=float)
    return {
        "n": len(arr),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


# Prepare temporary DB for timing so the real database is not modified
shutil.copyfile("database/embeddings.pkl", TEMP_DB_PATH)
original_db_path = enroll_module.DB_PATH
enroll_module.DB_PATH = TEMP_DB_PATH

# Enrollment timing
rows_enroll = [r for r in split_rows if r["split"] == "enrollment"][:N_ENROLL]
enroll_times = []
for i, row in enumerate(rows_enroll):
    user_id = f"timing_user_{i}_{row['identity']}"
    t0 = time.perf_counter()
    enroll(row["image_path"], user_id)
    t1 = time.perf_counter()
    enroll_times.append(t1 - t0)

# Warm-up verification / identification so model load is not counted heavily
warm_row = test_rows[0]
warm_claimed = int(warm_row["claimed_identity"]) if warm_row["claimed_identity"].isdigit() else warm_row["claimed_identity"]
try:
    verify(warm_row["image_path"], warm_claimed, threshold=THRESHOLD)
except Exception:
    pass
try:
    identify(warm_row["image_path"])
except Exception:
    pass

# Verification timing
verify_rows = test_rows[:N_VERIFY]
verify_times = []
for row in verify_rows:
    claimed = int(row["claimed_identity"]) if row["claimed_identity"].isdigit() else row["claimed_identity"]
    t0 = time.perf_counter()
    try:
        verify(row["image_path"], claimed, threshold=THRESHOLD)
    except Exception:
        pass
    t1 = time.perf_counter()
    verify_times.append(t1 - t0)

# Identification timing (1:N over current DB size)
identify_rows = test_rows[:N_IDENTIFY]
identify_times = []
for row in identify_rows:
    t0 = time.perf_counter()
    try:
        identify(row["image_path"])
    except Exception:
        pass
    t1 = time.perf_counter()
    identify_times.append(t1 - t0)

# Restore original DB path
enroll_module.DB_PATH = original_db_path

enroll_stats = summarize(enroll_times)
verify_stats = summarize(verify_times)
identify_stats = summarize(identify_times)
current_db_size = len(load_temp_db())

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["operation", "sample_idx", "time_s"])
    for i, t in enumerate(enroll_times):
        writer.writerow(["enroll", i, t])
    for i, t in enumerate(verify_times):
        writer.writerow(["verify", i, t])
    for i, t in enumerate(identify_times):
        writer.writerow(["identify_1N", i, t])

with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["operation", "n", "db_size", "mean_s", "median_s", "std_s", "min_s", "max_s"])
    writer.writerow(["enroll", enroll_stats["n"], current_db_size, enroll_stats["mean_s"], enroll_stats["median_s"], enroll_stats["std_s"], enroll_stats["min_s"], enroll_stats["max_s"]])
    writer.writerow(["verify", verify_stats["n"], current_db_size, verify_stats["mean_s"], verify_stats["median_s"], verify_stats["std_s"], verify_stats["min_s"], verify_stats["max_s"]])
    writer.writerow(["identify_1N", identify_stats["n"], current_db_size, identify_stats["mean_s"], identify_stats["median_s"], identify_stats["std_s"], identify_stats["min_s"], identify_stats["max_s"]])

print("Saved", OUT_CSV)
print("Saved", SUMMARY_CSV)
print()
print(f"Temporary DB size during timing: {current_db_size}")
print(f"Enroll    mean={enroll_stats['mean_s']:.4f}s  median={enroll_stats['median_s']:.4f}s  min={enroll_stats['min_s']:.4f}s  max={enroll_stats['max_s']:.4f}s")
print(f"Verify    mean={verify_stats['mean_s']:.4f}s  median={verify_stats['median_s']:.4f}s  min={verify_stats['min_s']:.4f}s  max={verify_stats['max_s']:.4f}s")
print(f"Identify  mean={identify_stats['mean_s']:.4f}s  median={identify_stats['median_s']:.4f}s  min={identify_stats['min_s']:.4f}s  max={identify_stats['max_s']:.4f}s")
