import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import time, os

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("  STEP 5 (FAST v2): Graph Feature Engineering")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/5] Loading data...")
df = pd.read_parquet("data/reviews_labelled.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
df = df.dropna(subset=["timestamp"])
print(f"  Reviews loaded: {len(df):,}  |  Unique reviewers: {df['user_id'].nunique():,}")

# ── 2. Build co-occurrence graph (OPTIMISED) ──────────────────
print("\n[2/5] Building reviewer co-occurrence graph...")
print("  Strategy: top 500 products only, 7-day window")
print("  (Reduced from 2000 to avoid timeout — still captures farm clusters)")

top_products = df["asin"].value_counts().head(500).index
df_top = df[df["asin"].isin(top_products)][["asin","user_id","timestamp"]].copy()
df_top = df_top.sort_values(["asin","timestamp"])
print(f"  Reviews in top 500 products: {len(df_top):,}")

G = nx.Graph()
G.add_nodes_from(df["user_id"].unique())

t0 = time.time()
edge_weights = defaultdict(int)
WINDOW_DAYS = 7

for asin, grp in df_top.groupby("asin"):
    users = grp[["user_id","timestamp"]].values
    n = len(users)
    if n > 500:          # cap per-product to avoid O(n²) explosion
        users = users[:500]
        n = 500
    for i in range(n):
        for j in range(i+1, n):
            try:
                days = (users[j][1] - users[i][1]).days
            except Exception:
                days = int((users[j][1] - users[i][1]) / np.timedelta64(1,'D'))
            if days > WINDOW_DAYS:
                break
            u, v = users[i][0], users[j][0]
            if u != v:
                edge_weights[(min(u,v), max(u,v))] += 1

elapsed = time.time() - t0
print(f"  Edge pairs computed in {elapsed:.1f}s  |  Unique edges: {len(edge_weights):,}")

for (u,v), w in edge_weights.items():
    G.add_edge(u, v, weight=w)

print(f"  Graph: {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges")

# ── 3. Extract graph features ─────────────────────────────────
print("\n[3/5] Extracting graph features (no clustering coeff)...")

print("  degree centrality...")
deg_cent = nx.degree_centrality(G)

print("  PageRank (max_iter=30)...")
try:
    pagerank = nx.pagerank(G, max_iter=30, tol=1e-3)
except Exception:
    pagerank = {n: 1/G.number_of_nodes() for n in G.nodes()}

print("  Connected component sizes...")
comp_size = {}
for comp in nx.connected_components(G):
    s = len(comp)
    for node in comp:
        comp_size[node] = s

print("  Weighted degree...")
weighted_deg = dict(G.degree(weight="weight"))

# ── 4. Build feature DataFrame ────────────────────────────────
print("\n[4/5] Assembling graph feature DataFrame...")
all_users = df["user_id"].unique()

graph_features = pd.DataFrame({
    "user_id":            all_users,
    "degree_centrality":  [deg_cent.get(u, 0)              for u in all_users],
    "pagerank_score":     [pagerank.get(u, 0)               for u in all_users],
    "log_component_size": [np.log1p(comp_size.get(u, 1))   for u in all_users],
    "weighted_degree":    [weighted_deg.get(u, 0)           for u in all_users],
})
print(f"  Graph features shape: {graph_features.shape}")

# ── 5. Merge + sanity check + save ────────────────────────────
print("\n[5/5] Merging with behaviour/NLP features and saving...")
feat  = pd.read_parquet("data/features.parquet")
labels = df.groupby("user_id")["is_fake"].max().reset_index()

merged = (feat
          .merge(graph_features, on="user_id", how="left")
          .merge(labels,         on="user_id", how="left"))

print(f"  Full feature matrix shape: {merged.shape}")
print(f"  Label distribution — FAKE: {(merged.is_fake==1).sum():,}  REAL: {(merged.is_fake==0).sum():,}")

print("\n── Graph feature gap (FAKE vs REAL) ──────────────────────")
for col in ["degree_centrality","pagerank_score","log_component_size","weighted_degree"]:
    fk = merged.loc[merged.is_fake==1, col].mean()
    rl = merged.loc[merged.is_fake==0, col].mean()
    ratio = fk/rl if rl > 0 else float("inf")
    print(f"  {col:25s}  FAKE: {fk:.5f}  REAL: {rl:.5f}  ratio: {ratio:.2f}x")

graph_features.to_parquet("data/graph_features.parquet",      index=False)
merged.to_parquet(        "data/features_with_graph.parquet", index=False)

print("\n  Saved: data/graph_features.parquet")
print("  Saved: data/features_with_graph.parquet  ← used in Step 6")
print("\n" + "=" * 60)
print("  STEP 5 COMPLETE.")
print("  Next → Step 6: Retrain XGBoost with graph features")
print("=" * 60)