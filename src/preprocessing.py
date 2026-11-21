import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/raw/digits.csv"
OUTPUT_DIR = "data/processed"
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "digits_processed.csv")
PARQUET_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "digits_features.parquet")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Input dataset not found at '{INPUT_PATH}'.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_PATH)

feature_cols = [c for c in df.columns if c.startswith("pixel_")]
if not feature_cols:
    raise ValueError("No feature columns starting with 'pixel_' were found in the dataset.")

# Ensure pixel columns are numeric and handle missing values before scaling
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
if df[feature_cols].isna().any().any():
    df[feature_cols] = df[feature_cols].fillna(0.0)

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Parse event_timestamp safely (coerce invalid values to NaT and normalize to UTC for Feast)
if "event_timestamp" not in df.columns:
    raise KeyError("Missing required 'event_timestamp' column.")

df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
if df["event_timestamp"].isna().any():
    bad_rows = int(df["event_timestamp"].isna().sum())
    raise ValueError(f"Found {bad_rows} rows with invalid or missing 'event_timestamp' values.")

# Save processed CSV and Parquet file for Feast offline store
df.to_csv(CSV_OUTPUT_PATH, index=False)

try:
    df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
except ImportError as exc:
    raise ImportError(
        "Parquet export requires 'pyarrow' or 'fastparquet'. Install with: pip install pyarrow"
    ) from exc

print("Preprocessing complete. Saved CSV and Parquet.")