"""
STEP 2 — Label Creation (Fake Review Heuristics)
Amazon Fake Review Network Detector
Amazon ML School Project 2026

Strategy: We have no ground-truth labels, so we build them from
3 well-established heuristics used in published fake review research.
Each heuristic catches a different type of fraud behaviour.
A review is labelled FAKE if it triggers ANY heuristic.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: Label Creation — Fake Review Heuristics")
print("="*60)

print("\n[1/6] Loading parquet files...")
df   = pd.read_parquet("data/reviews.parquet")
meta = pd.read_parquet("data/meta.parquet")
print(f"  Reviews loaded: {len(df):,}")
print(f"  Meta loaded:    {len(meta):,}")

# ── Normalise timestamp to seconds ────────────────────────────────────────────
# Amazon timestamps are in milliseconds in this dataset
df["timestamp"] = df["timestamp"] / 1000
df["review_date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

# ── HEURISTIC 1: Burst Reviewers ──────────────────────────────────────────────
# A reviewer who posts 5+ reviews within any 48-hour window is suspicious.
# Real customers rarely buy and review 5+ different products in 2 days.
print("\n[2/6] Heuristic 1 — Burst reviewers (5+ reviews in 48 hrs)...")

df_sorted = df.sort_values(["user_id", "timestamp"])

def is_burst_reviewer(group):
    """Return True for ALL reviews by a user if they ever had a burst."""
    if len(group) < 5:
        return pd.Series(False, index=group.index)
    timestamps = group["timestamp"].values
    for i in range(len(timestamps)):
        window = timestamps[(timestamps >= timestamps[i]) &
                            (timestamps <= timestamps[i] + 48 * 3600)]
        if len(window) >= 5:
            return pd.Series(True, index=group.index)
    return pd.Series(False, index=group.index)

burst_flags = df_sorted.groupby("user_id", group_keys=False).apply(is_burst_reviewer)
df["flag_burst"] = burst_flags.reindex(df.index).fillna(False)

n_burst = df["flag_burst"].sum()
print(f"  Flagged by burst heuristic: {n_burst:,} ({n_burst/len(df)*100:.1f}%)")

# ── HEURISTIC 2: Unverified 5-Star Reviews ────────────────────────────────────
# Unverified purchase + 5-star rating is the most common fake review pattern.
# Sellers buy reviews from farms — the reviewer never bought the product.
print("\n[3/6] Heuristic 2 — Unverified purchase + 5-star rating...")

df["flag_unverified_5star"] = (
    (df["verified_purchase"] == False) &
    (df["rating"] == 5.0)
)

n_unverified = df["flag_unverified_5star"].sum()
print(f"  Flagged by unverified 5-star: {n_unverified:,} ({n_unverified/len(df)*100:.1f}%)")

# ── HEURISTIC 3: High Text Similarity ─────────────────────────────────────────
# Review farms often reuse templates. Reviews on the SAME product with
# TF-IDF cosine similarity > 0.80 are almost certainly copy-paste fakes.
print("\n[4/6] Heuristic 3 — High text similarity on same product (cosine > 0.80)...")
print("  (Processing top products by review count — takes ~60 seconds)")

# Work on top 3000 most-reviewed products (captures most farm activity)
top_products = df["parent_asin"].value_counts().head(3000).index
df_top = df[df["parent_asin"].isin(top_products)].copy()

similar_review_ids = set()
vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=2)

products_processed = 0
for asin, group in df_top.groupby("parent_asin"):
    if len(group) < 3:
        continue
    texts = group["text"].fillna("").astype(str).tolist()
    try:
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
        np.fill_diagonal(sim_matrix, 0)  # ignore self-similarity
        # Flag any review with cosine similarity > 0.80 to another on same product
        flagged_mask = (sim_matrix > 0.80).any(axis=1)
        flagged_indices = group.index[flagged_mask].tolist()
        similar_review_ids.update(flagged_indices)
    except Exception:
        continue
    products_processed += 1

df["flag_similar_text"] = df.index.isin(similar_review_ids)
n_similar = df["flag_similar_text"].sum()
print(f"  Products processed: {products_processed:,}")
print(f"  Flagged by text similarity: {n_similar:,} ({n_similar/len(df)*100:.1f}%)")

# ── COMBINE: Final label ───────────────────────────────────────────────────────
# A review is FAKE (label=1) if it triggers ANY of the 3 heuristics
print("\n[5/6] Combining heuristics into final labels...")

df["is_fake"] = (
    df["flag_burst"] |
    df["flag_unverified_5star"] |
    df["flag_similar_text"]
).astype(int)

# ── Label statistics ───────────────────────────────────────────────────────────
total       = len(df)
n_fake      = df["is_fake"].sum()
n_real      = total - n_fake
fake_pct    = n_fake / total * 100

print(f"\n── Label distribution ─────────────────────────────")
print(f"  Total reviews:  {total:,}")
print(f"  FAKE  (label=1): {n_fake:,}  ({fake_pct:.1f}%)")
print(f"  REAL  (label=0): {n_real:,}  ({100-fake_pct:.1f}%)")
print(f"\n  Class imbalance ratio: 1 : {n_real/n_fake:.1f}")

print(f"\n── Heuristic breakdown ────────────────────────────")
print(f"  Burst only:           {(df['flag_burst'] & ~df['flag_unverified_5star'] & ~df['flag_similar_text']).sum():,}")
print(f"  Unverified 5★ only:   {(~df['flag_burst'] & df['flag_unverified_5star'] & ~df['flag_similar_text']).sum():,}")
print(f"  Similar text only:    {(~df['flag_burst'] & ~df['flag_unverified_5star'] & df['flag_similar_text']).sum():,}")
print(f"  Multiple flags:       {((df['flag_burst'].astype(int) + df['flag_unverified_5star'].astype(int) + df['flag_similar_text'].astype(int)) >= 2).sum():,}")

# ── Rating distribution by label ──────────────────────────────────────────────
print(f"\n── Rating distribution: FAKE vs REAL ──────────────")
rating_dist = df.groupby(["is_fake", "rating"]).size().unstack(fill_value=0)
rating_dist.index = ["REAL", "FAKE"]
print(rating_dist)

# ── Verified purchase by label ─────────────────────────────────────────────────
print(f"\n── Verified purchase rate ──────────────────────────")
vp = df.groupby("is_fake")["verified_purchase"].mean()
print(f"  REAL reviews — verified: {vp[0]*100:.1f}%")
print(f"  FAKE reviews — verified: {vp[1]*100:.1f}%")

# ── Save labelled dataset ──────────────────────────────────────────────────────
print("\n[6/6] Saving labelled dataset...")
df.to_parquet("data/reviews_labelled.parquet", index=False)
print(f"  Saved: data/reviews_labelled.parquet")

print("\n" + "="*60)
print("  STEP 2 COMPLETE. Labels created and saved.")
print("  Next → Step 3: Feature Engineering")
print("="*60 + "\n")