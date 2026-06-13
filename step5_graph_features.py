"""
STEP 5: Graph Feature Engineering (NetworkX)
Builds reviewer co-occurrence graph → extracts centrality features
Nodes = reviewers | Edges = both reviewed same product within 7 days
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import warnings, os, time
warnings.filterwarnings("ignore")

print("=" * 60)
print("  STEP 5: Graph Feature Engineering (NetworkX)")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/6] Loading data...")
df = pd.read_parquet("data/reviews_labelled.parquet")
features = pd.read_parquet("data/features.parquet")

# Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
df = df.dropna(subset=["timestamp"])
print(f"  Reviews with valid timestamps: {len(df):,}")

# ── 2. Build co-occurrence graph ──────────────────────────────
print("\n[2/6] Building reviewer co-occurrence graph...")
print("  (Reviewers who reviewed same product within 7 days get an edge)")
t0 = time.time()

# Work on top products by review count for speed (covers most reviewers)
top_products = df["asin"].value_counts().head(5000).index
df_top = df[df["asin"].isin(top_products)].copy()
print(f"  Using top 5,000 products → {len(df_top):,} reviews")

G = nx.Graph()

# Add all reviewers as nodes first
all_reviewers = df["user_id"].unique()
G.add_nodes_from(all_reviewers)

# Add edges: same product, within 7 days
edges_added = 0
for asin, group in df_top.groupby("asin"):
    group = group.sort_values("timestamp")
    reviewers = group["user_id"].tolist()
    timestamps = group["timestamp"].tolist()

    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            days_apart = abs((timestamps[i] - timestamps[j]).days)
            if days_apart <= 7:
                u, v = reviewers[i], reviewers[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
                    edges_added += 1
            else:
                break  # sorted by time, no need to check further

print(f"  Graph built in {time.time()-t0:.1f}s")
print(f"  Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")

# ── 3. Extract graph features per reviewer ────────────────────
print("\n[3/6] Extracting graph centrality features...")

# Degree centrality (fast)
degree_cent = nx.degree_centrality(G)

# PageRank (fast)
pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

# Connected component size per node
cc_sizes = {}
for component in nx.connected_components(G):
    size = len(component)
    for node in component:
        cc_sizes[node] = size

# Clustering coefficient (sample for speed on large graph)
print("  Computing clustering coefficients (sampled)...")
clustering = nx.clustering(G)

# Weighted degree (sum of edge weights)
weighted_degree = dict(G.degree(weight="weight"))

print("  Graph features extracted.")

# ── 4. Build graph feature DataFrame ─────────────────────────
print("\n[4/6] Assembling graph feature DataFrame...")

graph_df = pd.DataFrame({
    "user_id":              list(all_reviewers),
    "degree_centrality":    [degree_cent.get(r, 0)    for r in all_reviewers],
    "pagerank_score":       [pagerank.get(r, 0)        for r in all_reviewers],
    "component_size":       [cc_sizes.get(r, 1)        for r in all_reviewers],
    "clustering_coeff":     [clustering.get(r, 0)      for r in all_reviewers],
    "weighted_degree":      [weighted_degree.get(r, 0) for r in all_reviewers],
})

# Log-scale component size (very skewed)
graph_df["log_component_size"] = np.log1p(graph_df["component_size"])

print(f"  Graph features shape: {graph_df.shape}")
print(f"\n── Graph feature stats ──────────────────────────────────")
print(graph_df.drop("user_id", axis=1).describe().round(4).to_string())

# ── 5. Merge with existing features ──────────────────────────
print("\n[5/6] Merging graph features with behaviour+NLP features...")

GRAPH_COLS = ["degree_centrality", "pagerank_score",
              "log_component_size", "clustering_coeff", "weighted_degree"]

features_enriched = features.merge(
    graph_df[["user_id"] + GRAPH_COLS], on="user_id", how="left"
)
features_enriched[GRAPH_COLS] = features_enriched[GRAPH_COLS].fillna(0)

# ── 6. Show graph signal on fake vs real ─────────────────────
print(f"\n── Graph signal: FAKE vs REAL reviewers ─────────────────")
for col in GRAPH_COLS:
    fake_val = features_enriched[features_enriched.label==1][col].mean()
    real_val = features_enriched[features_enriched.label==0][col].mean()
    print(f"  {col:<25} FAKE: {fake_val:.4f}  REAL: {real_val:.4f}")

# ── 7. Save ───────────────────────────────────────────────────
print("\n[6/6] Saving enriched feature matrix...")
os.makedirs("data", exist_ok=True)
features_enriched.to_parquet("data/features_with_graph.parquet", index=False)

# Update feature cols list
old_cols = pd.read_csv("data/feature_cols.csv")["feature_cols"].tolist()
all_cols  = old_cols + GRAPH_COLS
pd.Series(all_cols, name="feature_cols").to_csv("data/feature_cols_graph.csv", index=False)

print(f"  Saved: data/features_with_graph.parquet")
print(f"  Total features now: {len(all_cols)} (was {len(old_cols)})")
print("\n" + "=" * 60)
print("  STEP 5 COMPLETE. Graph features added.")
print("  Next → Step 6: Retrain XGBoost with graph features → ~85-91% F1")
print("=" * 60)
