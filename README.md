# ReviewGuard — Fake Review Network Detector
Amazon ML School Project 2026

This repository implements ReviewGuard, a pipeline and demo app for detecting coordinated fake-review networks in product review datasets (Electronics). The approach combines reviewer-level behaviour & NLP signals with graph/network features (co-review graphs) and an XGBoost classifier, plus SHAP explanations and an interactive Streamlit demo (`app.py`).

## Contents

- `app.py` — Streamlit dashboard for exploration and scorer
- `data/` — input data and intermediate parquet files (large raw files are not committed)
- `models/` — exported model artifacts (`final_xgb.json`, `baseline_xgb.json`)
- `step1_load_data.py` … `step6_final_model.py` — pipeline scripts for reproducible stages
- `requirements.txt` — Python dependencies
- `.streamlit/` — Streamlit configuration
- `DEPLOY.md` — deployment notes and hosting considerations

## Quick start (local)
Recommended: use the Windows Python launcher `py` if `python` is not on your PATH.

1) Create a venv and install dependencies

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1    # or: .\.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m pip install pyarrow
```

2) (Optional) Verify environment

```powershell
py -m pip show streamlit
py -m pip show xgboost
py step1_smoke_test.py
```

3) Prepare data

- Small generated/intermediate files in `data/` may be included for convenience.
- Large raw datasets are NOT committed to the repo (they exceed GitHub limits). Use `step1_load_data.py` to download and build the input files required by the pipeline. Example:

```powershell
py step1_load_data.py
# This script downloads and writes data into the `data/` folder (parquet files)
```

If you already have the original Amazon dataset, place the files under `data/` as the scripts expect (see the top of `step1_load_data.py` for exact filenames).

4) Run the Streamlit demo

```powershell
py -m streamlit run app.py
# or specify a port
py -m streamlit run app.py --server.port 8502
```

Open `http://localhost:8501` (or the port you chose) to interact with the dashboard.

Live demo (hosted): https://fake-review-detector-amazon-gcqhpoivprdwps8vxhnjt6.streamlit.app/

## Reproducing the pipeline (high level)

The repository is organised as sequential steps. Run them in order to reproduce the feature engineering and model training:

1. `step1_load_data.py` — download / prepare raw data
2. `step2_labels.py` — label heuristics and initial labelling
3. `step3_features.py` — compute reviewer-level behaviour & NLP features
4. `step5_graph_features.py` — compute co-review network features
5. `step4_baseline_model.py` / `step6_final_model.py` — train XGBoost models and calibrate threshold

Each script writes intermediate parquet files into `data/`. If a step fails, re-run from that step.

## Models & explanation

- Model files live in `models/` (e.g. `models/final_xgb.json`).
- SHAP is used to explain predictions; the Streamlit app displays SHAP bar charts and feature breakdowns.

## GitHub & large files (important)

Two dataset files in `data/` (raw Amazon dumps) are very large and exceed GitHub's 100 MB limit. They must not be pushed to a standard Git remote. Recommended options:

- Use Git LFS to store large artifacts and keep your repo history clean. Example:

```powershell
py -m pip install git-lfs
git lfs install
git lfs track "data/*.gz"
git add .gitattributes
git commit -m "Track large data with LFS"
```

- If large files were accidentally committed, remove them from history (see `DEPLOY.md` or run `git filter-repo` / `bfg`), then force-push a cleaned branch.

This repository's `.gitignore` is configured to exclude raw large blobs (e.g. `data/*.gz`, `data/*.parquet`, `models/`) — do not add large raw datasets to the repo.

## Deployment notes

See `DEPLOY.md` for instructions about hosting the Streamlit app, using a VM, or Docker packaging.

## Troubleshooting

- Streamlit command not found: use `py -m streamlit run app.py` on Windows.
- If the app shows an old version: save files and restart Streamlit, or clear the Streamlit cache: `py -m streamlit cache clear`.
- Git push rejected (GH001): you have files >100MB in history. Remove them with `git filter-repo` or `bfg`, or move them to Git LFS, then force-push.

## Notes for maintainers

- Keep large datasets out of the repo; provide script-driven downloads instead.
- Keep `requirements.txt` up to date when adding libraries to the pipeline.

## License & attribution

This project is an educational reproduction for Amazon ML School. Check the original dataset licensing where appropriate (McAuley Lab / Amazon review datasets).

---
If you want, I can also add a tiny `README_DATA.md` describing exact filenames and a download checklist for reproducibility.

