
import os
import pickle
import numpy as np
from deepface import DeepFace

DB_PATH = "database/embeddings.pkl"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    os.makedirs("database", exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

def enroll(image_path, user_id):
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        embedding = np.array(result[0]["embedding"])
        db = load_db()
        db[user_id] = embedding
        save_db(db)
        print(f"Enrolled '{user_id}' successfully.")
        return True
    except Exception as e:
        print(f"Failed to enroll '{user_id}': {e}")
        return False

def get_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name="ArcFace",
        detector_backend="opencv",
        enforce_detection=True
    )
    return np.array(result[0]["embedding"])

if __name__ == "__main__":
    enroll("test_photos/me.jpg", "Bartosz Trams")