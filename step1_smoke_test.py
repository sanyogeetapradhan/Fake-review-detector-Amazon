"""
STEP 1 — Smoke Test (runs without internet, verifies all imports + logic)
Run this first to confirm your environment is working.
Then run step1_load_data.py to download the real dataset.
"""

import pandas as pd
import numpy as np
import json, gzip, os
from datetime import datetime, timedelta
import random

print("Testing imports...")
import sklearn, xgboost, networkx, shap, optuna
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
print("  All imports OK\n")

# ── Generate synthetic Amazon-style review data ────────────────────────────
random.seed(42)
np.random.seed(42)

N = 2000  # synthetic reviews

user_ids    = [f"USER_{i:05d}" for i in np.random.randint(0, 400, N)]
product_ids = [f"ASIN_{i:05d}" for i in np.random.randint(0, 200, N)]
base_date   = datetime(2023, 1, 1)

fake_texts = [
    "Absolutely amazing product!! Best purchase ever!!!",
    "5 stars!! Love love love this item!! Highly recommend!!!",
    "Perfect!! Works exactly as described!! Great quality!!!",
    "Wow amazing!! So happy with this purchase!! 5 stars!!",
]
real_texts = [
    "Works fine for my use case. Battery life is decent.",
    "Pretty good product. Had a minor issue with the setup but sorted it out.",
    "Does what it says. Build quality could be better for the price.",
    "Good value. Shipping was fast. Would buy again.",
]

reviews = []
for i in range(N):
    is_fake   = random.random() < 0.12  # 12% fake — realistic class imbalance
    timestamp = base_date + timedelta(days=random.randint(0, 365))
    reviews.append({
        "user_id":          user_ids[i],
        "asin":             product_ids[i],
        "rating":           5.0 if is_fake else float(random.choice([3, 4, 4, 5])),
        "text":             random.choice(fake_texts) if is_fake else random.choice(real_texts),
        "verified_purchase": False if is_fake else random.random() > 0.3,
        "timestamp":        int(timestamp.timestamp() * 1000),
        "helpful_vote":     0 if is_fake else random.randint(0, 20),
        "_synthetic_label": is_fake,  # ground truth for smoke test only
    })

df = pd.DataFrame(reviews)

print(f"Synthetic dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Fake review rate: {df['_synthetic_label'].mean():.1%}")
print(f"Rating dist:\n{df['rating'].value_counts().sort_index()}\n")

# ── Test VADER sentiment ───────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()
df["sentiment_compound"] = df["text"].apply(
    lambda t: analyzer.polarity_scores(t)["compound"]
)
print(f"Avg sentiment (fake):  {df[df['_synthetic_label']]['sentiment_compound'].mean():.3f}")
print(f"Avg sentiment (real):  {df[~df['_synthetic_label']]['sentiment_compound'].mean():.3f}")

# ── Test textstat readability ──────────────────────────────────────────────
df["readability"] = df["text"].apply(textstat.flesch_reading_ease)
print(f"\nAvg readability (fake): {df[df['_synthetic_label']]['readability'].mean():.1f}")
print(f"Avg readability (real): {df[~df['_synthetic_label']]['readability'].mean():.1f}")

# ── Test NetworkX graph construction ──────────────────────────────────────
import networkx as nx
G = nx.Graph()
product_reviewers = df.groupby("asin")["user_id"].apply(list)
for asin, reviewers in product_reviewers.items():
    for i in range(len(reviewers)):
        for j in range(i+1, len(reviewers)):
            if G.has_edge(reviewers[i], reviewers[j]):
                G[reviewers[i]][reviewers[j]]["weight"] += 1
            else:
                G.add_edge(reviewers[i], reviewers[j], weight=1)

print(f"\nReviewer graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Test XGBoost ──────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

df["exclamation_density"] = df["text"].apply(lambda t: t.count("!") / max(len(t), 1))
df["review_length"]       = df["text"].apply(len)
df["is_verified"]         = df["verified_purchase"].astype(int)

features = ["rating", "sentiment_compound", "readability", "exclamation_density",
            "review_length", "is_verified"]
X = df[features]
y = df["_synthetic_label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                      scale_pos_weight=(y==0).sum()/(y==1).sum())
model.fit(X_train, y_train)
preds = model.predict(X_test)
f1  = f1_score(y_test, preds)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

print(f"\nSmoke-test XGBoost on synthetic data:")
print(f"  F1:      {f1:.3f}")
print(f"  AUC-ROC: {auc:.3f}")

# Save for next step
os.makedirs("data", exist_ok=True)
df.to_parquet("data/smoke_test_reviews.parquet", index=False)
print("\n  Saved smoke test data to data/smoke_test_reviews.parquet")

print("\n" + "="*55)
print("  SMOKE TEST PASSED. Your environment is fully ready.")
print("  Next: run step1_load_data.py to download real data.")
print("="*55)
