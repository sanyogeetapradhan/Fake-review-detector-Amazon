import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

print("=" * 60)
print("  STEP 6: Final Model — Graph + NLP + Behaviour + SMOTE + Optuna")
print("=" * 60)

# ── 1. Load full feature matrix ───────────────────────────────
print("\n[1/6] Loading features_with_graph.parquet...")
df = pd.read_parquet("data/features_with_graph.parquet")

# Drop leaky heuristic flags and non-feature cols
DROP_COLS = ["user_id", "is_fake", "label", "any_burst", "any_unverified_5star", "any_similar_text"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].fillna(0)
y = df["is_fake"].astype(int)

print(f"  Features: {X.shape[1]}  |  Reviewers: {len(X):,}")
print(f"  FAKE: {y.sum():,} ({y.mean()*100:.1f}%)  REAL: {(y==0).sum():,}")
print(f"  Feature list: {list(X.columns)}")

# ── 2. Train/test split ───────────────────────────────────────
print("\n[2/6] Splitting 80/20 stratified...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. SMOTE — fix class imbalance ───────────────────────────
print("\n[3/6] Applying SMOTE to training set...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE — FAKE: {y_train_sm.sum():,}  REAL: {(y_train_sm==0).sum():,}")

# ── 4. Optuna hyperparameter tuning ──────────────────────────
print("\n[4/6] Optuna tuning (40 trials)...")

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3),
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_sm, y_train_sm,
                             cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, show_progress_bar=False)
best_params = study.best_params
best_params.update({"tree_method": "hist", "eval_metric": "logloss", "random_state": 42})
print(f"  Best CV F1:   {study.best_value:.4f}")
print(f"  Best params:  {best_params}")

# ── 5. Train final model ──────────────────────────────────────
print("\n[5/6] Training final model on full SMOTE training set...")
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train_sm, y_train_sm)

y_pred  = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)

print("\n── FINAL MODEL RESULTS ──────────────────────────────────")
print(f"  F1 Score  : {f1:.4f}  ({f1*100:.1f}%)")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\n── Confusion Matrix ──────────────────────────────────────")
print(f"  True Neg  (REAL→REAL): {cm[0,0]:,}")
print(f"  False Pos (REAL→FAKE): {cm[0,1]:,}")
print(f"  False Neg (FAKE→REAL): {cm[1,0]:,}")
print(f"  True Pos  (FAKE→FAKE): {cm[1,1]:,}")
print(f"\n── Classification Report ─────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))

# ── Results comparison table ──────────────────────────────────
print("── Accuracy progression (full story) ────────────────────")
results = pd.DataFrame([
    {"Model": "Baseline XGBoost (behaviour only)",  "F1": 0.704, "ROC-AUC": 0.944},
    {"Model": "Final Model (+ graph + SMOTE + Optuna)", "F1": round(f1,3), "ROC-AUC": round(auc,3)},
])
print(results.to_string(index=False))

# Save results
results.to_csv("data/results_table.csv", index=False)
print("\n  Saved: data/results_table.csv")

# ── 6. SHAP feature importance ────────────────────────────────
print("\n[6/6] Generating SHAP feature importance plot...")
explainer  = shap.TreeExplainer(final_model)
shap_vals  = explainer.shap_values(X_test)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False,
                  max_display=20, color="#e84118")
plt.title("Feature Importance — Fake Review Detector\n(SHAP values, higher = more important)", 
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/shap_final.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/shap_final.png")

# Save model
final_model.save_model("models/final_xgb.json")
print("  Saved: models/final_xgb.json")

print("\n" + "=" * 60)
print("  STEP 6 COMPLETE.")
print(f"  Final F1: {f1*100:.1f}%  |  Baseline: 70.4%  |  Gain: +{(f1-0.704)*100:.1f}pp")
print("  Next → Step 7: Pyvis interactive graph visualisation")
print("=" * 60)