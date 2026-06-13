import pandas as pd
import json, gzip, os
from tqdm import tqdm

print("Re-saving meta with price fix...")

records = []
with gzip.open("data/Electronics_meta.jsonl.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f)):
        if i >= 100_000:
            break
        try:
            records.append(json.loads(line))
        except:
            continue

meta = pd.DataFrame(records)
meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
nested = ["images","videos","features","description","categories","details","bought_together"]
meta_clean = meta.drop(columns=[c for c in nested if c in meta.columns])
meta_clean.to_parquet("data/meta.parquet", index=False)

# Also fix reviews (drop images column)
print("Re-saving reviews with images column dropped...")
df = pd.read_parquet("data/reviews.parquet") if os.path.exists("data/reviews.parquet") else None
if df is None:
    import json, gzip
    records2 = []
    with gzip.open("data/Electronics_reviews.jsonl.gz", "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if i >= 500_000: break
            try: records2.append(json.loads(line))
            except: continue
    df = pd.DataFrame(records2)

df = df.drop(columns=["images"], errors="ignore")
df.to_parquet("data/reviews.parquet", index=False)

print(f"Done! reviews.parquet: {len(df):,} rows | meta.parquet: {len(meta_clean):,} rows")
print("STEP 1 COMPLETE. Ready for Step 2.")