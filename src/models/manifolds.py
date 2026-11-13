"""Data preparation and cached embedding experiments for notebook 05.

These are descriptive embeddings: scaling deliberately uses the full dataset,
matching the original notebook, rather than a predictive train/test workflow.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.config import CLEAN_DATA_PATH, RESULTS_DIR

NOTEBOOK_RESULTS = RESULTS_DIR / "05_manifolds_tsne_umap_embeddings"
FEATURES = (
    "Days for shipping (real)", "Days for shipment (scheduled)",
    "Benefit per order", "Sales per customer", "Order Item Discount",
    "Order Item Discount Rate", "Order Item Profit Ratio", "Order Item Quantity",
    "Sales", "Product Price",
)
SEEDS = (0, 21, 42, 123, 7, 13, 57, 99, 202, 314)
KS = (2, 5, 10, 30, 50)


def read_embedding(path, index, method):
    """Require the same observation IDs/order and two finite coordinates."""
    frame = pd.read_csv(path, index_col="source_index")
    columns = [f"{'t-SNE' if method == 'tsne' else 'UMAP'} {i}" for i in (1, 2)]
    if not frame.index.is_unique or not frame.index.equals(index):
        raise ValueError(f"Saved embedding rows do not match input: {path}")
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Missing coordinate columns: {path}")
    frame = frame[columns]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Nonfinite embedding coordinates: {path}")
    return frame


class ManifoldData:
    """Original data, standardized features, and proportional stratified sample."""

    def __init__(self, path=CLEAN_DATA_PATH, features=FEATURES, frame=None):
        self.frame = pd.read_csv(path) if frame is None else frame.copy(deep=True)
        self.features = list(features)
        self.X = self.frame[self.features].copy()
        if not self.frame.index.is_unique or not np.isfinite(self.X.to_numpy(dtype=float)).all():
            raise ValueError("Data require unique row IDs and finite numerical features.")
        self.scaler = StandardScaler()
        self.scaled = pd.DataFrame(self.scaler.fit_transform(self.X),
                                   columns=self.features, index=self.X.index)

    def standardization_summary(self):
        from IPython.display import display
        display(self.scaled.head())
        print("Means:")
        display(self.scaled.mean().round(4))
        print("Standard deviations:")
        display(self.scaled.std(ddof=0).round(4))

    def sample(self, size=30000, random_state=42):
        if not 1 <= size <= len(self.frame):
            raise ValueError("Sample size must be between 1 and the dataset size.")
        strata = (self.frame["Customer Segment"].astype(str) + " | "
                  + self.frame["Department Name"].astype(str))
        rng = np.random.default_rng(random_state)
        shuffled = (pd.DataFrame({"stratum": strata})
                    .assign(random_order=rng.random(len(self.frame))).sort_values("random_order"))
        counts = shuffled["stratum"].value_counts()
        expected = counts / counts.sum() * size
        allocation = np.floor(expected).astype(int)
        for group in (expected - allocation).nlargest(size - allocation.sum()).index:
            allocation.loc[group] += 1
        indices = []
        for group, count in allocation.items():
            indices.extend(shuffled.index[shuffled["stratum"] == group][:count])
        self.sample_scaled = self.scaled.loc[indices].copy()
        self.sample_frame = self.frame.loc[indices].copy()
        return self

    def sample_summary(self):
        print("Original observations:", len(self.frame))
        print("Sample observations:", len(self.sample_scaled))
        print("Features:", self.sample_scaled.shape[1])
        return (self.sample_frame.groupby(["Customer Segment", "Department Name"])
                .size().reset_index(name="Count"))

    def duplicate_summary(self):
        groups = self.scaled.groupby(list(self.scaled.columns), dropna=False).size().sort_values(ascending=False)
        duplicates = int(self.scaled.duplicated().sum())
        print(f"Dataset shape: {self.scaled.shape}")
        print(f"Total observations: {len(self.scaled):,}")
        print(f"Unique feature vectors: {len(self.scaled)-duplicates:,}")
        print(f"Duplicate observations: {duplicates:,}")
        print(f"Duplicate percentage: {100*duplicates/len(self.scaled):.2f}%")
        print(f"Groups with repeated feature vectors: {(groups > 1).sum():,}")
        print(f"Largest duplicate group: {groups.max():,}")
        return groups.head(10)


class EmbeddingSearch:
    """Fit missing configurations; validate and reuse existing coordinate CSVs."""

    def __init__(self, method, configurations, random_state=42,
                 force_retrain=False, results_dir=NOTEBOOK_RESULTS):
        if method not in ("tsne", "umap"):
            raise ValueError("method must be tsne or umap")
        self.method, self.configurations = method, list(configurations)
        self.random_state, self.force_retrain = random_state, force_retrain
        self.results_dir = Path(results_dir)

    def path_for(self, config, seed=None):
        seed = self.random_state if seed is None else seed
        if self.method == "tsne":
            name = f"tsne_perplexity_{config}_seed_{seed}.csv"
        else:
            neighbors, distance = config
            name = f"umap_neighbors_{neighbors}_mindist_{distance}_seed_{seed}.csv"
        return self.results_dir / name

    def estimator(self, config, seed):
        if self.method == "tsne":
            return TSNE(n_components=2, perplexity=config, learning_rate="auto",
                        init="random", random_state=seed, n_jobs=-1)
        import umap
        neighbors, distance = config
        return umap.UMAP(n_components=2, n_neighbors=neighbors, min_dist=distance,
                         metric="euclidean", random_state=seed)

    def fit_one(self, X, config, seed=None, path=None):
        seed = self.random_state if seed is None else seed
        path = self.path_for(config, seed) if path is None else Path(path)
        if path.exists() and not self.force_retrain:
            frame = read_embedding(path, X.index, self.method)
            print(f"Loaded {path.name}. Existing embedding preserved.", flush=True)
            return frame
        path.parent.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=1):
            values = self.estimator(config, seed).fit_transform(X)
        columns = [f"{'t-SNE' if self.method == 'tsne' else 'UMAP'} {i}" for i in (1, 2)]
        frame = pd.DataFrame(values, index=X.index, columns=columns)
        if not np.isfinite(values).all():
            raise ValueError("Embedding contains nonfinite coordinates.")
        frame.to_csv(path, index_label="source_index", mode="w" if self.force_retrain else "x")
        print(f"Completed and saved {path.name}.", flush=True)
        return frame

    def run(self, data):
        self.data = data
        self.embeddings, self.files = {}, {}
        for config in self.configurations:
            self.files[config] = self.path_for(config)
            self.embeddings[config] = self.fit_one(data.sample_scaled, config).to_numpy()
        return self


class TSNESearch(EmbeddingSearch):
    def __init__(self, perplexities=(5, 15, 30, 50), **kwargs):
        super().__init__("tsne", perplexities, **kwargs)


class UMAPSearch(EmbeddingSearch):
    def __init__(self, n_neighbors=(10, 30, 50, 100), min_dist=(0.0, 0.1, 0.5), **kwargs):
        super().__init__("umap", product(n_neighbors, min_dist), **kwargs)


class FullEmbedding:
    """A full-data embedding kept separate from sample/seed experiments."""

    def __init__(self, method, perplexity=30, n_neighbors=30, min_dist=0.1,
                 random_state=42, force_retrain=False, results_dir=NOTEBOOK_RESULTS):
        self.method = method
        self.config = perplexity if method == "tsne" else (n_neighbors, min_dist)
        self.search = EmbeddingSearch(method, [self.config], random_state,
                                      force_retrain, results_dir)
        self.path = Path(results_dir) / f"{method}_final_full_dataset.csv"

    def fit(self, data):
        self.data = data
        self.frame = self.search.fit_one(data.scaled, self.config, path=self.path)
        print(f"Final embedding shape: {self.frame.shape}")
        print(f"File: {self.path}")
        return self
