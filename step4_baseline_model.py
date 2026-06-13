"""
STEP 4: XGBoost Baseline Model
Trains on behaviour + NLP features → establishes ~74-81% F1 baseline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings, os
warnings.filterwarnings("ignore")

print("=" * 60)
print("  STEP 4: XGBoost Baseline Model")
print("=" * 60)

# ── 1. Load features ──────────────────────────────────────────
print("\n[1/5] Loading feature matrix...")
features = pd.read_parquet("data/features.parquet")
feature_cols = pd.read_csv("data/feature_cols.csv")["feature_cols"].tolist()

X = features[feature_cols].values
y = features["label"].values
print(f"  X shape: {X.shape}  |  Fake: {y.sum():,} ({y.mean()*100:.1f}%)")

# ── 2. Train/test split (stratified) ─────────────────────────
print("\n[2/5] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. Train XGBoost ──────────────────────────────────────────
print("\n[3/5] Training XGBoost (default params — baseline)...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos_weight,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
    n_jobs            = -1,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── 4. Evaluate ───────────────────────────────────────────────
print("\n[4/5] Evaluating model...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

f1       = f1_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)
cm       = confusion_matrix(y_test, y_pred)

# 5-fold CV F1
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1", n_jobs=-1)

print(f"\n── Baseline Results ─────────────────────────────────────")
print(f"  F1 Score       : {f1:.4f}  ({f1*100:.1f}%)")
print(f"  ROC-AUC        : {roc_auc:.4f}")
print(f"  5-Fold CV F1   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n── Confusion Matrix ─────────────────────────────────────")
print(f"  True Neg  (REAL→REAL): {cm[0,0]:,}")
print(f"  False Pos (REAL→FAKE): {cm[0,1]:,}")
print(f"  False Neg (FAKE→REAL): {cm[1,0]:,}")
print(f"  True Pos  (FAKE→FAKE): {cm[1,1]:,}")
print(f"\n── Full Classification Report ───────────────────────────")
print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))

# ── 5. Save model + SHAP plot ─────────────────────────────────
print("[5/5] Saving model and SHAP feature importance...")
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

model.save_model("models/baseline_xgb.json")

# SHAP summary plot
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:2000])  # sample for speed

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test[:2000],
                  feature_names=feature_cols,
                  plot_type="bar", show=False)
plt.title("Feature Importance (SHAP) — Baseline Model", fontsize=14)
plt.tight_layout()
plt.savefig("plots/shap_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: models/baseline_xgb.json")
print(f"  Saved: plots/shap_baseline.png")

# Save baseline score for comparison table later
pd.DataFrame([{
    "model":   "Baseline XGBoost (Behaviour + NLP)",
    "f1":      round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "cv_f1":   round(cv_scores.mean(), 4),
}]).to_csv("data/results_table.csv", index=False)

print("\n" + "=" * 60)
print("  STEP 4 COMPLETE. Baseline model trained and saved.")
print(f"  Baseline F1: {f1*100:.1f}%")
print("  Next → Step 5: Graph features (NetworkX) → push to ~88-91%")
print("=" * 60)
