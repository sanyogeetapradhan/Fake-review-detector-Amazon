import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import optuna, shap, warnings, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

print("=" * 60)
print("  STEP 6b: Accuracy Boost — Threshold Tuning + Scale-pos-weight")
print("=" * 60)

# ── 1. Load ───────────────────────────────────────────────────
print("\n[1/5] Loading features...")
df = pd.read_parquet("data/features_with_graph.parquet")

DROP = ["user_id","is_fake","label","any_burst","any_unverified_5star","any_similar_text"]
feature_cols = [c for c in df.columns if c not in DROP]
X = df[feature_cols].fillna(0)
y = df["is_fake"].astype(int)
print(f"  Features: {len(feature_cols)}  |  Reviewers: {len(X):,}")
print(f"  FAKE: {y.sum():,} ({y.mean()*100:.1f}%)  REAL: {(y==0).sum():,}")

# ── 2. Split ──────────────────────────────────────────────────
print("\n[2/5] Splitting 80/20 stratified...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 3. Optuna — tune on RAW (imbalanced) train, score on val ──
print("\n[3/5] Optuna tuning (50 trials) on raw imbalanced train...")
scale = (y_train==0).sum() / (y_train==1).sum()   # ~4.7
print(f"  scale_pos_weight = {scale:.2f}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

def objective(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 300, 700),
        max_depth        = trial.suggest_int("max_depth", 3, 9),
        learning_rate    = trial.suggest_float("lr", 0.01, 0.2, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample", 0.5, 1.0),
        min_child_weight = trial.suggest_int("mcw", 1, 20),
        gamma            = trial.suggest_float("gamma", 0, 3),
        reg_alpha        = trial.suggest_float("alpha", 0, 2),
        reg_lambda       = trial.suggest_float("lambda", 0.5, 4),
        scale_pos_weight = trial.suggest_float("spw", scale*0.5, scale*2.0),
        tree_method="hist", random_state=42,
    )
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    proba = m.predict_proba(X_val)[:,1]
    # find best threshold on val
    best_f1 = 0
    for thr in np.arange(0.2, 0.8, 0.05):
        f1 = f1_score(y_val, (proba >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)
best = study.best_params
print(f"  Best val F1 (threshold-tuned): {study.best_value:.4f}")

# ── 4. Train final model + find best threshold ────────────────
print("\n[4/5] Training final model + threshold search...")
final_params = dict(
    n_estimators     = best["n_estimators"],
    max_depth        = best["max_depth"],
    learning_rate    = best["lr"],
    subsample        = best["subsample"],
    colsample_bytree = best["colsample"],
    min_child_weight = best["mcw"],
    gamma            = best["gamma"],
    reg_alpha        = best["alpha"],
    reg_lambda       = best["lambda"],
    scale_pos_weight = best["spw"],
    tree_method="hist", random_state=42,
)
model = xgb.XGBClassifier(**final_params)
model.fit(X_train, y_train)

# Find best threshold on val split
val_proba = model.predict_proba(X_val)[:,1]
best_thr, best_val_f1 = 0.5, 0
for thr in np.arange(0.15, 0.85, 0.01):
    f1 = f1_score(y_val, (val_proba >= thr).astype(int))
    if f1 > best_val_f1:
        best_val_f1, best_thr = f1, thr
print(f"  Best threshold: {best_thr:.2f}  (val F1={best_val_f1:.4f})")

# Evaluate on test
test_proba = model.predict_proba(X_test)[:,1]
y_pred = (test_proba >= best_thr).astype(int)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, test_proba)
cm  = confusion_matrix(y_test, y_pred)

print("\n── FINAL RESULTS ────────────────────────────────────────")
print(f"  F1 Score  : {f1:.4f}  ({f1*100:.1f}%)")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\n── Confusion Matrix ──────────────────────────────────────")
print(f"  True Neg  (REAL→REAL): {cm[0,0]:,}")
print(f"  False Pos (REAL→FAKE): {cm[0,1]:,}")
print(f"  False Neg (FAKE→REAL): {cm[1,0]:,}")
print(f"  True Pos  (FAKE→FAKE): {cm[1,1]:,}")
print(f"\n{classification_report(y_test, y_pred, target_names=['REAL','FAKE'])}")

print("── Accuracy Progression ─────────────────────────────────")
results = pd.DataFrame([
    {"Stage": "1. Baseline (behaviour only)",        "F1": "70.4%", "ROC-AUC": "0.944"},
    {"Stage": "2. + Graph features",                 "F1": "73.8%", "ROC-AUC": "0.941"},
    {"Stage": "3. + scale_pos_weight + threshold",   "F1": f"{f1*100:.1f}%", "ROC-AUC": f"{auc:.3f}"},
])
print(results.to_string(index=False))
results.to_csv("data/results_table.csv", index=False)

# ── 5. SHAP ───────────────────────────────────────────────────
print("\n[5/5] SHAP feature importance...")
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_test)
plt.figure(figsize=(10,7))
shap.summary_plot(sv, X_test, plot_type="bar", show=False, max_display=20, color="#e84118")
plt.title("Feature Importance — Fake Review Detector\n(SHAP, all 20 features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/shap_final.png", dpi=150, bbox_inches="tight")
plt.close()

model.save_model("models/final_xgb.json")
print("  Saved: plots/shap_final.png")
print("  Saved: models/final_xgb.json")
print("  Saved: data/results_table.csv")

print("\n" + "=" * 60)
print(f"  STEP 6b COMPLETE. Final F1: {f1*100:.1f}% | Baseline: 70.4%")
print("  Next → Step 7: Pyvis interactive graph visualisation")
print("=" * 60)