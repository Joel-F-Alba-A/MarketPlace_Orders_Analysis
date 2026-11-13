"""Resumable structure-preservation metrics and cross-seed neighborhood stability."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.models.manifolds import KS, SEEDS, NOTEBOOK_RESULTS, EmbeddingSearch, read_embedding


def nearest_neighbors(X, k):
    print(f"Finding {k} neighbors for {len(X):,} observations...", flush=True)
    return NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X).kneighbors(return_distance=False)


def chunked_trustworthiness(X, embedded_neighbors, ks, working_memory=32):
    """Rank original distances once per chunk for all pending k values."""
    n = len(X)
    penalties = {k: 0 for k in ks}
    start = 0
    started = last_report = time.monotonic()
    for distances in pairwise_distances_chunked(
            X, metric="euclidean", working_memory=working_memory, n_jobs=1):
        rows = len(distances)
        row_indices = np.arange(rows)
        distances[row_indices, start + row_indices] = np.inf
        order = np.argsort(distances, axis=1)
        inverse_ranks = np.empty_like(order)
        inverse_ranks[row_indices[:, None], order] = np.arange(1, n + 1)
        for k in ks:
            neighbors = embedded_neighbors[start:start + rows, :k]
            excess = inverse_ranks[row_indices[:, None], neighbors] - k
            penalties[k] += int(excess[excess > 0].sum())
        start += rows
        now = time.monotonic()
        if now - last_report >= 10 or start == n:
            elapsed = now - started
            print(f"Trustworthiness k={ks}: {start:,}/{n:,} ({100*start/n:.1f}%) | "
                  f"elapsed {elapsed/60:.1f} min | ETA {elapsed*(n-start)/start/60:.1f} min",
                  flush=True)
            last_report = now
    return {k: 1 - 2 * penalties[k] / (n * k * (2 * n - 3 * k - 1)) for k in ks}


@njit(cache=True)
def mean_neighbor_overlap(sorted_a, sorted_b):
    """Mean intersection fraction for equally sized, sorted neighbor-ID rows."""
    total = 0
    n, k = sorted_a.shape
    for row in range(n):
        i = j = 0
        while i < k and j < k:
            a, b = sorted_a[row, i], sorted_b[row, j]
            if a == b:
                total += 1
                i += 1
                j += 1
            elif a < b:
                i += 1
            else:
                j += 1
    return total / (n * k)


class PreservationEvaluation:
    """Original-space fidelity; preserve the notebook's existing cache identities."""

    def __init__(self, ks=KS, evaluation_size=30000, random_state=42, working_memory=32):
        self.ks = list(ks)
        self.evaluation_size, self.random_state = evaluation_size, random_state
        self.working_memory = working_memory

    @classmethod
    def from_csv(cls, path, random_state=42):
        """Load a completed table for presentation without preparing or fitting data."""
        path = Path(path)
        results = pd.read_csv(path)
        ks = [int(c.split("@")[1]) for c in results if c.startswith("Trustworthiness@")]
        evaluation = cls(ks=ks, random_state=random_state)
        evaluation.method = "tsne" if "Perplexity" in results else "umap"
        evaluation.path, evaluation.results = path, results
        names = ["Perplexity"] if evaluation.method == "tsne" else ["n_neighbors", "min_dist"]
        evaluation.comparison = pd.DataFrame([
            {**{name: row[name] for name in names}, "k": k,
             "Trustworthiness": row[f"Trustworthiness@{k}"],
             "Neighborhood Overlap": row[f"Neighborhood Overlap@{k}"]}
            for _, row in results.iterrows() for k in ks])
        return evaluation

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        self._results.to_csv(temporary)
        temporary.replace(self.path)

    def run(self, search):
        self.method, self.search = search.method, search
        sample = search.data.sample_scaled
        n = self.evaluation_size
        if not sample.index.is_unique or not 1 < n <= len(sample):
            raise ValueError("Evaluation requires unique IDs and 1 < size <= sample size.")
        if not self.ks or not all(isinstance(k, int) and 1 <= k < n/2 for k in self.ks):
            raise ValueError("Every k must be an integer with 1 <= k < evaluation_size/2.")
        evaluation = sample.sample(n=n, random_state=self.random_state)
        X = evaluation.to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite evaluation features.")
        fingerprint = hashlib.sha256(X.tobytes())
        fingerprint.update(pd.util.hash_pandas_object(evaluation.index).to_numpy().tobytes())
        # Legacy identities allow completed metric tables to load without rerunning.
        version = "multi-k-v1" if self.method == "tsne" else "umap-multi-k-v1"
        fingerprint.update(repr((tuple(self.ks), self.random_state,
                                 search.configurations, version)).encode())
        for path in search.files.values():
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024*1024), b""):
                    fingerprint.update(chunk)
        self.path = search.results_dir / (
            f"{self.method}_preservation_metrics_eval{n}_ks{'-'.join(map(str,self.ks))}_"
            f"seed{self.random_state}_{fingerprint.hexdigest()[:12]}.csv")
        columns = [name for k in self.ks for name in
                   (f"Trustworthiness@{k}", f"Neighborhood Overlap@{k}")] + ["Global Spearman"]
        names = ["Perplexity"] if self.method == "tsne" else ["n_neighbors", "min_dist"]
        index = (pd.Index(search.configurations, name=names[0]) if self.method == "tsne"
                 else pd.MultiIndex.from_tuples(search.configurations, names=names))
        if self.path.exists():
            results = pd.read_csv(self.path).set_index(names)
            if not results.index.is_unique:
                raise ValueError("Duplicate configurations in metric cache.")
            results = results.reindex(index=index, columns=columns)
        else:
            results = pd.DataFrame(index=index, columns=columns, dtype=float)
        self._results = results
        print(f"Progress file: {self.path}", flush=True)
        with threadpool_limits(limits=1):
            original_neighbors = None
            distance_indices = np.random.default_rng(self.random_state).choice(
                len(X), size=min(1000,len(X)), replace=False)
            original_distances = None
            for number, config in enumerate(search.configurations, 1):
                if results.loc[config].notna().all():
                    print(f"Skipping {config}: already complete.", flush=True)
                    continue
                print(f"[{number}/{len(index)}] Evaluating {config}", flush=True)
                frame = read_embedding(search.files[config], sample.index, self.method)
                Y = frame.loc[evaluation.index].to_numpy(dtype=float)
                if pd.isna(results.loc[config, "Global Spearman"]):
                    if original_distances is None:
                        original_distances = pdist(X[distance_indices])
                    results.loc[config, "Global Spearman"] = float(spearmanr(
                        original_distances, pdist(Y[distance_indices])).statistic)
                    self._save()
                overlaps = [k for k in self.ks if pd.isna(results.loc[config,f"Neighborhood Overlap@{k}"])]
                trusts = [k for k in self.ks if pd.isna(results.loc[config,f"Trustworthiness@{k}"])]
                if overlaps or trusts:
                    embedded = nearest_neighbors(Y, max(self.ks))
                if overlaps:
                    if original_neighbors is None:
                        original_neighbors = nearest_neighbors(X, max(self.ks))
                    for k in overlaps:
                        value = mean_neighbor_overlap(np.sort(original_neighbors[:,:k],axis=1),
                                                       np.sort(embedded[:,:k],axis=1))
                        results.loc[config,f"Neighborhood Overlap@{k}"] = value
                        self._save()
                        print(f"Saved Neighborhood Overlap@{k}: {value:.6f}", flush=True)
                if trusts:
                    for k,value in chunked_trustworthiness(X,embedded,trusts,self.working_memory).items():
                        results.loc[config,f"Trustworthiness@{k}"] = value
                        print(f"Saved Trustworthiness@{k}: {value:.6f}", flush=True)
                    self._save()
        self.results = results.reset_index()
        self.comparison = pd.DataFrame([
            {**{name: row[name] for name in names}, "k": k,
             "Trustworthiness": row[f"Trustworthiness@{k}"],
             "Neighborhood Overlap": row[f"Neighborhood Overlap@{k}"]}
            for _,row in self.results.iterrows() for k in self.ks])
        return self

    def display(self):
        from IPython.display import display
        display(self.results.round(6))
        display(self.comparison.round(6))


class SeedStability:
    """Fit/reuse embeddings, then compare observation-ID neighborhoods across seeds.

    UMAP runs include a versioned manifest. Set force_retrain=True to force a new
    timestamped experiment; default loading avoids expensive accidental reruns.
    """

    def __init__(self, method, seeds=SEEDS, ks=KS, perplexity=30,
                 n_neighbors=30, min_dist=0.1, force_retrain=False,
                 results_dir=NOTEBOOK_RESULTS):
        if method not in ("tsne", "umap"):
            raise ValueError("method must be tsne or umap")
        self.method, self.seeds, self.ks = method, list(seeds), list(ks)
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError("Provide at least two distinct seeds.")
        self.config = perplexity if method == "tsne" else (n_neighbors,min_dist)
        self.force_retrain, self.results_dir = force_retrain, Path(results_dir)

    def fit(self, data):
        self.data = data
        X = data.sample_scaled
        if not X.index.is_unique or not self.ks or not all(1 <= k < len(X) for k in self.ks):
            raise ValueError("Unique row IDs and 1 <= k < sample size are required.")
        self.embeddings = {}
        if self.method == "tsne":
            search = EmbeddingSearch("tsne", [self.config], force_retrain=self.force_retrain,
                                      results_dir=self.results_dir)
            self.run_dir = self.results_dir / "tsne_seed_stability"
            self.run_dir.mkdir(parents=True,exist_ok=True)
            for seed in self.seeds:
                self.embeddings[seed] = search.fit_one(X,self.config,seed).to_numpy()
            return self
        import umap
        values = X.to_numpy(dtype=np.float32,copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite UMAP input.")
        h = hashlib.sha256(values.tobytes())
        h.update(pd.util.hash_pandas_object(X.index).to_numpy().tobytes())
        parameters = dict(n_components=2,n_neighbors=self.config[0],min_dist=self.config[1],
                          metric="euclidean",init="spectral",n_epochs=200,low_memory=True,n_jobs=1)
        expected = dict(umap_version=umap.__version__,sample_size=len(X),features=list(X.columns),
                        input_sha256=h.hexdigest(),seeds=self.seeds,evaluation_ks=self.ks,
                        parameters=parameters)
        parent = self.results_dir / "umap_seed_stability_runs"
        if not self.force_retrain:
            for manifest_path in sorted(parent.glob("*/manifest.json"),reverse=True):
                manifest = json.loads(manifest_path.read_text())
                if all(manifest.get(k)==v for k,v in expected.items()) and manifest.get("completed_seeds")==self.seeds:
                    self.run_dir = manifest_path.parent
                    for seed in self.seeds:
                        self.embeddings[seed] = read_embedding(
                            self.run_dir/f"embedding_seed_{seed}.csv",X.index,"umap").to_numpy()
                    print(f"Loaded validated UMAP seed run: {self.run_dir}",flush=True)
                    return self
        self.run_dir = parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir.mkdir(parents=True,exist_ok=False)
        manifest = dict(expected,completed_seeds=[])
        manifest_path = self.run_dir/"manifest.json"
        manifest_path.write_text(json.dumps(manifest,indent=2))
        with threadpool_limits(limits=1):
            for number,seed in enumerate(self.seeds,1):
                started = time.monotonic()
                print(f"[{number}/{len(self.seeds)}] Fitting UMAP seed {seed}",flush=True)
                coordinates = umap.UMAP(**parameters,random_state=seed,verbose=True).fit_transform(values.copy())
                if coordinates.shape != (len(X),2) or not np.isfinite(coordinates).all():
                    raise ValueError(f"Invalid UMAP coordinates for seed {seed}.")
                pd.DataFrame(coordinates,index=X.index,columns=["UMAP 1","UMAP 2"]).to_csv(
                    self.run_dir/f"embedding_seed_{seed}.csv",index_label="source_index",mode="x")
                self.embeddings[seed] = coordinates
                manifest["completed_seeds"].append(seed)
                manifest_path.write_text(json.dumps(manifest,indent=2))
                print(f"Seed {seed} completed in {time.monotonic()-started:.1f}s",flush=True)
        return self

    def run(self):
        n = len(self.data.sample_scaled)
        with threadpool_limits(limits=1):
            neighbors = {seed: nearest_neighbors(values,max(self.ks))
                         for seed,values in self.embeddings.items()}
        self.matrices, rows = {}, []
        for k in self.ks:
            sorted_neighbors = {seed: np.sort(values[:,:k],axis=1) for seed,values in neighbors.items()}
            matrix = pd.DataFrame(np.eye(len(self.seeds)),index=self.seeds,columns=self.seeds)
            for i,seed_a in enumerate(self.seeds):
                for seed_b in self.seeds[i+1:]:
                    value = mean_neighbor_overlap(sorted_neighbors[seed_a],sorted_neighbors[seed_b])
                    matrix.loc[seed_a,seed_b] = matrix.loc[seed_b,seed_a] = value
            upper = matrix.to_numpy()[np.triu_indices(len(self.seeds),1)]
            self.matrices[k] = matrix
            matrix.to_csv(self.run_dir/f"stability_k{k}.csv",index_label="seed")
            rows.append(dict(k=k,mean_overlap=upper.mean(),pairwise_std=upper.std(),
                             min_overlap=upper.min(),max_overlap=upper.max(),seed_pairs=len(upper),
                             random_overlap_baseline=k/(n-1)))
            print(f"Stability@{k}: mean={upper.mean():.4f}",flush=True)
        self.summary = pd.DataFrame(rows)
        self.summary.to_csv(self.run_dir/"summary.csv",index=False)
        return self

    def display_matrices(self):
        from IPython.display import display
        for k,matrix in self.matrices.items():
            print(f"Neighborhood Stability@{k}")
            display(matrix.round(3))
