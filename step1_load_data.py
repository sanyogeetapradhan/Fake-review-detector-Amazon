"""
STEP 1 — Data Download & Loading
Amazon Fake Review Network Detector
Amazon ML School Project 2026
"""

import os
import json
import gzip
import requests
import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Amazon Review Data 2023 (McAuley Lab, UCSD)
# We use Electronics — large enough to be impressive, small enough for 2 days
# Full dataset index: https://mcauleylab.ucsd.edu/data/amazon_2023/
REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"
META_URL    = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"

REVIEWS_PATH = os.path.join(DATA_DIR, "Electronics_reviews.jsonl.gz")
META_PATH    = os.path.join(DATA_DIR, "Electronics_meta.jsonl.gz")

# How many rows to load (set None to load ALL ~5GB — takes 30+ min)
# 500k rows is plenty for this project and loads in ~2 minutes
MAX_ROWS = 500_000


# ── DOWNLOAD HELPER ───────────────────────────────────────────────────────────
def download_file(url, dest_path):
    """Download a file with a progress bar. Skips if already downloaded."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1_000_000
        print(f"  Already downloaded: {dest_path} ({size_mb:.1f} MB)")
        return

    print(f"  Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved to {dest_path}")


# ── LOAD JSONL.GZ ─────────────────────────────────────────────────────────────
def load_jsonl_gz(path, max_rows=None):
    """Load a gzipped JSONL file into a DataFrame."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {os.path.basename(path)}")):
            if max_rows and i >= max_rows:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  STEP 1: Download & Load Amazon Review Data 2023")
    print("="*60)

    # 1. Download
    print("\n[1/4] Downloading datasets...")
    download_file(REVIEWS_URL, REVIEWS_PATH)
    download_file(META_URL, META_PATH)

    # 2. Load reviews
    print(f"\n[2/4] Loading reviews (first {MAX_ROWS:,} rows)...")
    df = load_jsonl_gz(REVIEWS_PATH, max_rows=MAX_ROWS)

    # 3. Load product metadata (first 100k — we only need category/title)
    print("\n[3/4] Loading product metadata...")
    meta = load_jsonl_gz(META_PATH, max_rows=100_000)

    # 4. Sanity checks
    print("\n[4/4] Sanity checks...")
    print("\n── Reviews DataFrame ──────────────────────────────")
    print(f"  Shape:   {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Sample row:")
    print(df.iloc[0].to_dict())

    print("\n── Metadata DataFrame ─────────────────────────────")
    print(f"  Shape:   {meta.shape}")
    print(f"  Columns: {list(meta.columns)}")

    print("\n── Key statistics ─────────────────────────────────")
    if "rating" in df.columns:
        print(f"  Rating distribution:\n{df['rating'].value_counts().sort_index()}")
    if "verified_purchase" in df.columns:
        print(f"\n  Verified purchase:\n{df['verified_purchase'].value_counts()}")
    if "user_id" in df.columns:
        print(f"\n  Unique reviewers:  {df['user_id'].nunique():,}")
    if "asin" in df.columns:
        print(f"  Unique products:   {df['asin'].nunique():,}")

    # 5. Save clean parquet for fast loading in later steps
    print("\n── Saving to parquet for fast reloading ───────────")
    df.to_parquet(os.path.join(DATA_DIR, "reviews.parquet"), index=False)
    meta.to_parquet(os.path.join(DATA_DIR, "meta.parquet"), index=False)
    print(f"  Saved: data/reviews.parquet")
    print(f"  Saved: data/meta.parquet")

    print("\n" + "="*60)
    print("  STEP 1 COMPLETE. Ready for Step 2: Label Creation.")
    print("="*60 + "\n")

    return df, meta


if __name__ == "__main__":
    df, meta = main()
