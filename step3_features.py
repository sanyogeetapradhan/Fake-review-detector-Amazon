"""
STEP 3: Feature Engineering
Builds 3 signal groups → merges into one model-ready feature matrix
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import re
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  STEP 3: Feature Engineering")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/6] Loading labelled data...")
df = pd.read_parquet("data/reviews_labelled.parquet")
reviewer_stats = pd.read_parquet("data/reviewer_stats.parquet")
print(f"  Reviews: {len(df):,}  |  Reviewers: {len(reviewer_stats):,}")

# ── 2. Reviewer-level behaviour features ─────────────────────
print("\n[2/6] Building reviewer behaviour features...")

rev = df.groupby("user_id").agg(
    total_reviews        = ("asin",             "count"),
    verified_purchase_ratio = ("verified_purchase", "mean"),
    avg_rating           = ("rating",           "mean"),
    rating_std           = ("rating",           "std"),
    unique_products      = ("asin",             "nunique"),
    pct_5star            = ("rating",           lambda x: (x == 5).mean()),
    pct_1star            = ("rating",           lambda x: (x == 1).mean()),
    any_burst            = ("flag_burst",        "max"),
    any_unverified_5star = ("flag_unverified_5star", "max"),
    any_similar_text     = ("flag_similar_text", "max"),
).reset_index()

# Global avg rating to compute deviation
global_avg = df["rating"].mean()
rev["avg_rating_deviation"] = abs(rev["avg_rating"] - global_avg)
rev["rating_std"] = rev["rating_std"].fillna(0)
rev["review_burstiness"] = rev["total_reviews"] / rev["unique_products"].clip(lower=1)

print(f"  Behaviour features built: {rev.shape[1]-1} features for {len(rev):,} reviewers")

# ── 3. NLP features per review ───────────────────────────────
print("\n[3/6] Computing NLP features per review (takes ~2 min)...")

sia = SentimentIntensityAnalyzer()

def nlp_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    sentiment   = sia.polarity_scores(text)["compound"]
    readability = textstat.flesch_reading_ease(text)
    excl_density = text.count("!") / max(len(text), 1) * 100
    caps_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    word_count  = len(text.split())
    return sentiment, readability, excl_density, caps_ratio, word_count

# Use 'text' column (may be called 'text' or 'reviewText')
text_col = "text" if "text" in df.columns else "reviewText"
if text_col not in df.columns:
    df["text"] = ""
    text_col = "text"

results = df[text_col].fillna("").apply(nlp_features)
df["sentiment"]     = results.apply(lambda x: x[0])
df["readability"]   = results.apply(lambda x: x[1])
df["excl_density"]  = results.apply(lambda x: x[2])
df["caps_ratio"]    = results.apply(lambda x: x[3])
df["word_count"]    = results.apply(lambda x: x[4])

print(f"  NLP features done.")
print(f"  Avg sentiment — FAKE: {df[df.is_fake==1]['sentiment'].mean():.3f}  "
      f"REAL: {df[df.is_fake==0]['sentiment'].mean():.3f}")
print(f"  Avg readability — FAKE: {df[df.is_fake==1]['readability'].mean():.1f}  "
      f"REAL: {df[df.is_fake==0]['readability'].mean():.1f}")

# ── 4. Reviewer-level NLP aggregates ─────────────────────────
print("\n[4/6] Aggregating NLP features to reviewer level...")

nlp_agg = df.groupby("user_id").agg(
    avg_sentiment    = ("sentiment",   "mean"),
    std_sentiment    = ("sentiment",   "std"),
    avg_readability  = ("readability", "mean"),
    avg_excl_density = ("excl_density","mean"),
    avg_caps_ratio   = ("caps_ratio",  "mean"),
    avg_word_count   = ("word_count",  "mean"),
    std_word_count   = ("word_count",  "std"),
).reset_index()
nlp_agg = nlp_agg.fillna(0)

# ── 5. Merge all features ─────────────────────────────────────
print("\n[5/6] Merging all features into one matrix...")

# Get one label per reviewer (majority vote)
reviewer_labels = df.groupby("user_id")["is_fake"].max().reset_index()
reviewer_labels.columns = ["user_id", "label"]

features = reviewer_labels \
    .merge(rev,     on="user_id", how="left") \
    .merge(nlp_agg, on="user_id", how="left")

features = features.fillna(0)

FEATURE_COLS = [
    "total_reviews", "verified_purchase_ratio", "avg_rating",
    "rating_std", "unique_products", "pct_5star", "pct_1star",
    "avg_rating_deviation", "review_burstiness",
    "any_burst", "any_unverified_5star", "any_similar_text",
    "avg_sentiment", "std_sentiment", "avg_readability",
    "avg_excl_density", "avg_caps_ratio",
    "avg_word_count", "std_word_count",
]

X = features[FEATURE_COLS]
y = features["label"]

print(f"  Feature matrix shape: {X.shape}")
print(f"  Label distribution: FAKE={y.sum():,} ({y.mean()*100:.1f}%)  REAL={(y==0).sum():,}")

# ── 6. Save ───────────────────────────────────────────────────
print("\n[6/6] Saving feature matrix...")
features[["user_id", "label"] + FEATURE_COLS].to_parquet("data/features.parquet", index=False)
pd.Series(FEATURE_COLS, name="feature_cols").to_csv("data/feature_cols.csv", index=False)

print(f"\n  Saved: data/features.parquet")
print(f"\n── Feature summary ──────────────────────────────────────")
print(X.describe().round(3).to_string())
print("\n" + "=" * 60)
print("  STEP 3 COMPLETE. Feature matrix saved.")
print("  Next → Step 4: Train XGBoost baseline (~74% F1)")
print("=" * 60)
