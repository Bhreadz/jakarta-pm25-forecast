from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
df = df.drop(columns=["boundary_layer_height", "lat", "lon"]).sort_values(["station", "date"])

year = df["date"].dt.year
df[year <= 2024].to_csv(ROOT / "data" / "train_2020_2024.csv", index=False)
df[year == 2025].to_csv(ROOT / "data" / "test_2025.csv", index=False)
print(f"train: {(year <= 2024).sum()} rows, test: {(year == 2025).sum()} rows")
