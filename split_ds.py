import random
from pathlib import Path
import pandas as pd

SUBSET_DIR = Path("celeba_subset").resolve()
IDENTITY_TXT = SUBSET_DIR / "identity_subset.txt"
ENROLLED_CSV = SUBSET_DIR / "enrolled_identities.csv"
UNKNOWN_CSV = SUBSET_DIR / "unknown_identities.csv"

OUT_DIR = Path("data")
OUT_CSV = "splits.csv"

SEED = 42


def load_identity_txt(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            image_name, identity = parts
            rows.append({
                "image_name": image_name,
                "identity": int(identity),
            })
    return pd.DataFrame(rows)


def rel_image_path(img_name: str) -> str:
    return (Path("img_align_celeba") / img_name).as_posix()


def main():
    random.seed(SEED)

    df = load_identity_txt(IDENTITY_TXT)
    enrolled_ids = set(pd.read_csv(ENROLLED_CSV)["identity"].tolist())
    unknown_ids = set(pd.read_csv(UNKNOWN_CSV)["identity"].tolist())

    rows = []

    for identity in sorted(enrolled_ids):
        person_df = df[df["identity"] == identity].copy()

        if len(person_df) < 8:
            raise ValueError(f"Identity {identity} ma mniej niż 8 zdjęć.")

        image_names = person_df["image_name"].tolist()
        random.shuffle(image_names)

        enrollment_img = image_names[0]
        n_genuine = random.choice([2, 3])
        genuine_imgs = image_names[1:1 + n_genuine]
        impostor_imgs = image_names[1 + n_genuine:]

        rows.append({
            "image_name": enrollment_img,
            "image_path": rel_image_path(enrollment_img),
            "identity": identity,
            "split": "enrollment",
            "identity_type": "enrolled",
        })

        for img in genuine_imgs:
            rows.append({
                "image_name": img,
                "image_path": rel_image_path(img),
                "identity": identity,
                "split": "genuine_test",
                "identity_type": "enrolled",
            })

        for img in impostor_imgs:
            rows.append({
                "image_name": img,
                "image_path": rel_image_path(img),
                "identity": identity,
                "split": "impostor_pool",
                "identity_type": "enrolled",
            })

    for identity in sorted(unknown_ids):
        person_df = df[df["identity"] == identity].copy()

        for img in person_df["image_name"].tolist():
            rows.append({
                "image_name": img,
                "image_path": rel_image_path(img),
                "identity": identity,
                "split": "unknown_user",
                "identity_type": "unknown",
            })

    splits_df = pd.DataFrame(rows)

    dupes = splits_df.groupby(["identity", "image_name"]).size().reset_index(name="count")
    if (dupes["count"] > 1).any():
        raise ValueError("To samo zdjęcie trafiło do więcej niż jednego splitu.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(OUT_CSV, index=False)

    print(f"Saved to: {OUT_CSV}")
    print(splits_df.head())


if __name__ == "__main__":
    main()