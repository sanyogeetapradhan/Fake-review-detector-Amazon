# Fake Review Network Detector
### Amazon ML School Project 2026

## Setup (run once)
```bash
pip install -r requirements.txt
pip install pyarrow
```

## Step 1 — Verify environment
```bash
python3 step1_smoke_test.py
```

## Step 1 — Download real data (~2 GB download, ~2 min)
```bash
python3 step1_load_data.py
```
Data saved to: `data/reviews.parquet` and `data/meta.parquet`

## Project structure
```
fake_review_detector/
├── requirements.txt
├── data/                    ← auto-created by scripts
├── step1_smoke_test.py      ← run first to verify env
├── step1_load_data.py       ← downloads real Amazon data
├── step2_labels.py          ← coming next
├── step3_features.py        ← coming next
├── step4_baseline.py        ← coming next
└── ...
```
