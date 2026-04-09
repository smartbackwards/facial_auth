from enroll import enroll
import pandas as pd

df = pd.read_csv("data/splits.csv")
enrollment_rows = df[df["split"] == "enrollment"]
for _, row in enrollment_rows.iterrows():
    enroll(row["image_path"], row["identity"])