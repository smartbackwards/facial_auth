import pickle
import numpy as np
from enroll import get_embedding

DB_PATH = "database/embeddings.pkl"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify(image_path, claimed_id, threshold=0.5):
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    if claimed_id not in db:
        return False, 0.0
    query_emb = get_embedding(image_path)
    enrolled_emb = db[claimed_id]
    score = cosine_similarity(query_emb, enrolled_emb)
    return score >= threshold, float(score)