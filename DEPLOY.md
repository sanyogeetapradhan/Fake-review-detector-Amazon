# Fake Review Network Detector — Deployment Guide

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (free, public URL in 2 min)

1. Push your project folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo → set Main file: `app.py`
4. Click Deploy

### Folder structure required:
```
your-repo/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   ├── reviews_labelled.parquet
│   ├── features_with_graph.parquet
│   └── graph_features.parquet
└── models/
    └── final_xgb.json
```

> Note: parquet files can be large. If GitHub rejects them (>100MB),
> use Git LFS: `git lfs track "*.parquet"`
