# Facial Recognition Auth System
**Biometria — Projekt 1**

A biometric face authentication system built with ArcFace embeddings. Supports user enrollment, 1:1 verification, and 1:N identification. Built and tested as part of a university biometrics course project.

---

## Project structure

```
facial_auth/
├── enroll.py          # enrollment pipeline + embedding extraction
├── verify.py          # verification & identification logic
├── metrics.py         # FAR, FRR, ROC, threshold sweep
├── tests/             # test scripts for noise, luminance, JPEG, etc.
├── database/
│   └── embeddings.pkl # enrolled user embeddings (not raw photos)
├── data/
│   ├── enrollment/    # one image per enrolled user
│   ├── test/          # test images (genuine + impostor)
│   └── splits.csv     # image_path, identity, split label
└── results/           # CSV outputs from each test run
```

---

## Setup

**Requirements:** Python 3.8+, Windows/macOS/Linux

```bash
pip install deepface opencv-python numpy scikit-learn matplotlib
```

Model weights (~500MB) are downloaded automatically on first run.

---

## Quickstart

**Enroll a user:**
```python
from enroll import enroll
enroll("path/to/photo.jpg", "user_id")
```

**Verify a user:**
```python
from verify import verify
match, score = verify("path/to/query.jpg", "claimed_user_id")
```

**Extract an embedding directly:**
```python
from enroll import get_embedding
embedding = get_embedding("path/to/photo.jpg")  # returns np.array of shape (512,)
```

---

## Model

- **Backbone:** ArcFace (via DeepFace)
- **Detector:** RetinaFace (fallback: MTCNN)
- **Embedding size:** 512 dimensions
- **Similarity metric:** Cosine similarity

---

## Data

- **Dataset:** FaceScrub / CelebA subset
- **Enrolled users:** 80 identities (including all group members)
- **Splits:** enrollment / genuine test / impostor test — strictly disjoint
- **Raw photos are never stored** — only embeddings are persisted in `database/`

---

## Tests

| # | Test |
|---|------|
| 1 | Baseline accuracy (500 images) |
| 2 | FAR/FRR vs threshold, ROC curve |
| 3 | Unknown users (+ 100 non-enrolled) | 
| 4 | Noise robustness (5 PSNR bands) |
| 5 | Luminance changes |
| 6 | Enrollment & verification timing | 
| 7 | JPEG compression (3 quality levels) |

---

## Authors

| Name | Index |
|------|-------|
| Bartek Trams | 268421 |
| Bartek Mila | — |
| Krzysiek Szydlowski | — |
