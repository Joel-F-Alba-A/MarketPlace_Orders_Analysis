"""Factor-analysis and ICA workflows for notebook 04, with resumable searches."""
from pathlib import Path
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.config import CLEAN_DATA_PATH, RESULTS_DIR, SEED
from .dimensionality_reduction import DEFAULT_FEATURES, relative_frobenius_error

NOTEBOOK_RESULTS = RESULTS_DIR / "04_recommendation_ica_matrix_factorization"


def save_table(table, path, index_label=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    table.to_csv(temporary, index=index_label is not None, index_label=index_label)
    temporary.replace(path)


class FactorData:
    """70/15/15 split and training-fitted standardization of numeric features."""

    def __init__(self, path=CLEAN_DATA_PATH, features=DEFAULT_FEATURES,
                 random_state=SEED, frame=None):
        self.frame = pd.read_csv(path) if frame is None else frame.copy()
        self.features = list(features)
        self.X = (self.frame[self.features].replace([np.inf, -np.inf], np.nan)
                  .dropna().astype(float))
        self.train, temporary = train_test_split(self.X, test_size=.3, random_state=random_state)
        self.validation, self.test = train_test_split(temporary, test_size=.5, random_state=random_state)
        self.scaler = StandardScaler().fit(self.train)
        for split in ("train", "validation", "test"):
            setattr(self, f"{split}_scaled", self.scaler.transform(getattr(self, split)))

    def split_summary(self):
        return pd.DataFrame([{"Dataset": s, "Observations": len(getattr(self, s)),
                              "Features": len(self.features)}
                             for s in ("train", "validation", "test")])


class FactorSearch:
    """Parallel FA/ICA component search; checkpoint every completed k."""

    def __init__(self, method, k_values=None, n_bootstraps=10, n_clusters=3,
                 max_iter=None, tol=None, random_state=SEED, n_jobs=-1,
                 force_retrain=False, results_dir=NOTEBOOK_RESULTS):
        if method not in ("fa", "ica"):
            raise ValueError("method must be fa or ica")
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be positive")
        self.method, self.k_values = method, k_values
        self.n_bootstraps, self.n_clusters = n_bootstraps, n_clusters
        self.max_iter = max_iter if max_iter is not None else (3000 if method == "fa" else 2000)
        self.tol = tol if tol is not None else (1e-3 if method == "fa" else 1e-4)
        self.random_state, self.n_jobs = random_state, n_jobs
        self.force_retrain = force_retrain
        self.results_dir = Path(results_dir) / method

    def estimator(self, k, seed=None):
        options = dict(n_components=k, random_state=self.random_state if seed is None else seed,
                       max_iter=self.max_iter, tol=self.tol)
        if self.method == "fa":
            return FactorAnalysis(rotation="varimax", **options)
        return FastICA(whiten="unit-variance", algorithm="parallel", fun="logcosh", **options)

    def coefficients(self, model):
        return model.components_ if self.method == "fa" else model.mixing_.T

    def reconstruct(self, model, scores):
        return (scores @ model.components_ + model.mean_ if self.method == "fa"
                else model.inverse_transform(scores))

    def _stability(self, X, k, reference):
        rng = np.random.default_rng(self.random_state)
        similarities = []
        for b in range(self.n_bootstraps):
            sample = X[rng.choice(len(X), len(X), replace=True)]
            model = self.estimator(k, self.random_state + b + 1).fit(sample)
            similarity = np.abs(cosine_similarity(reference, self.coefficients(model)))
            rows, cols = linear_sum_assignment(-similarity)
            similarities.append(similarity[rows, cols].mean())
        return np.mean(similarities), np.std(similarities)

    def _evaluate(self, k):
        checkpoint = self.cache_dir / f"{self.method}_k{k}_metrics.csv"
        model_path = self.cache_dir / f"{self.method}_k{k}.joblib"
        if not self.force_retrain and checkpoint.exists() and model_path.exists():
            return pd.read_csv(checkpoint).iloc[0].to_dict()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = self.estimator(k)
            X, V = self.data.train_scaled, self.data.validation_scaled
            Z = model.fit_transform(X)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            scores = model.transform(V)
            coefficients = self.coefficients(model)
            row = {"k": k}
            for name, original, latent in (("Train", X, Z), ("Validation", V, scores)):
                reconstructed = self.reconstruct(model, latent)
                error = relative_frobenius_error(original, reconstructed)
                row.update({f"{name} MSE": mean_squared_error(original, reconstructed),
                            f"{name} Relative Frobenius": error,
                            f"{name} Reconstructed Energy": 1 - error**2})
            row["Stability Mean"], row["Stability Std"] = self._stability(X, k, coefficients)
            squared = coefficients**2
            normalized = squared / np.maximum(squared.sum(axis=1, keepdims=True), 1e-12)
            row["Mean Concentration"] = np.mean(np.sum(normalized**2, axis=1))
            row["Silhouette"] = np.nan
            if k >= 2 and len(scores) > self.n_clusters:
                labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                                n_init=10).fit_predict(scores)
                if 1 < len(np.unique(labels)) < len(scores):
                    try:
                        row["Silhouette"] = silhouette_score(scores, labels,
                            sample_size=min(5000, len(scores)), random_state=self.random_state)
                    except ValueError:
                        pass
            if self.method == "ica":
                row["Mean Absolute Kurtosis"] = np.mean(np.abs(kurtosis(scores, axis=0, fisher=True, bias=False)))
            row.update(Converged=converged, Iterations=model.n_iter_)
            temporary = model_path.with_suffix(f".{os.getpid()}.tmp")
            joblib.dump(model, temporary)
            temporary.replace(model_path)
            save_table(pd.DataFrame([row]), checkpoint)
            return row

    def run(self, data):
        self.data = data
        # Keep the existing FA experiment identity so old checkpoints remain usable.
        settings = {"cache_version": 1, "random_state": self.random_state,
                    "n_bootstraps": self.n_bootstraps, "n_clusters": self.n_clusters,
                    "max_iter": self.max_iter, "tol": self.tol,
                    "rotation": "varimax" if self.method == "fa" else None,
                    "sklearn_version": sklearn.__version__, "features": list(data.train.columns)}
        self.cache_dir = self.results_dir / joblib.hash((settings, data.train_scaled, data.validation_scaled))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        values = list(range(1, min(data.train_scaled.shape))) if self.k_values is None else list(self.k_values)
        if not values or any(k < 1 or k > min(data.train_scaled.shape) for k in values):
            raise ValueError("Invalid component counts")
        rows, pending = [], []
        for k in values:
            checkpoint = self.cache_dir / f"{self.method}_k{k}_metrics.csv"
            if not self.force_retrain and checkpoint.exists() and (self.cache_dir / f"{self.method}_k{k}.joblib").exists():
                rows.append(pd.read_csv(checkpoint).iloc[0].to_dict())
            else:
                pending.append(k)
        def update():
            self.results = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
            self.results["k"] = self.results["k"].astype(int)
            save_table(self.results, self.cache_dir / f"{self.method}_results.csv")
            save_table(self.results, self.results_dir / f"{self.method}_results.csv")
        if rows:
            update()
        print(f"{self.method.upper()}: loaded {len(rows)}; remaining {len(pending)}")
        with Parallel(n_jobs=self.n_jobs, backend="loky", return_as="generator_unordered") as parallel:
            for row in parallel(delayed(self._evaluate)(k) for k in pending):
                rows.append(row)
                update()
                print(f"Saved {self.method.upper()} k={int(row['k'])}")
        return self


class FASearch(FactorSearch):
    def __init__(self, **kwargs):
        super().__init__("fa", **kwargs)


class ICASearch(FactorSearch):
    def __init__(self, **kwargs):
        super().__init__("ica", **kwargs)


class FactorAnalysisResult:
    """Selected model's coefficients, split metrics, and observation scores."""

    def __init__(self, search, k=5):
        self.search, self.k = search, k

    def fit(self):
        search = self.search
        path = search.cache_dir / f"{search.method}_k{self.k}.joblib"
        if path.exists():
            self.model = joblib.load(path)
        else:
            self.model = search.estimator(self.k).fit(search.data.train_scaled)
            joblib.dump(self.model, path)
        prefix = "Factor" if search.method == "fa" else "IC"
        names = [f"{prefix} {i+1}" for i in range(self.k)]
        self.loadings = pd.DataFrame(search.coefficients(self.model).T,
                                    index=search.data.features, columns=names)
        rows, self.scores = [], {}
        for split in ("train", "validation", "test"):
            X = getattr(search.data, f"{split}_scaled")
            latent = self.model.transform(X)
            self.scores[split] = pd.DataFrame(latent, index=getattr(search.data, split).index, columns=names)
            reconstructed = search.reconstruct(self.model, latent)
            error = relative_frobenius_error(X, reconstructed)
            rows.append({"Dataset": split.capitalize(), "MSE": mean_squared_error(X, reconstructed),
                         "Relative Frobenius": error, "Reconstructed Energy": 1-error**2})
        self.metrics = pd.DataFrame(rows)
        self.kurtosis = pd.DataFrame({"Component": names,
            "Excess Kurtosis": kurtosis(self.scores["validation"], axis=0, fisher=True, bias=False)})
        self.kurtosis["Absolute Kurtosis"] = self.kurtosis["Excess Kurtosis"].abs()
        save_table(self.loadings, search.cache_dir / f"{search.method}_k{self.k}_loadings.csv", "Feature")
        save_table(self.metrics, search.cache_dir / f"{search.method}_k{self.k}_reconstruction.csv")
        return self


class FactorComparison:
    """Compare selected FA/ICA models with PCA tables exported by notebook 03."""

    def __init__(self, fa, ica, pca_k=5, pca_dir=None):
        self.fa, self.ica, self.pca_k = fa, ica, pca_k
        folder = Path(pca_dir) if pca_dir else RESULTS_DIR / "03_svd_pca_probabilistic_pca_lsa"
        self.pca_loadings = pd.read_csv(folder / f"pca_k{pca_k}_loadings.csv", index_col="Feature")
        self.pca_metrics = pd.read_csv(folder / f"pca_k{pca_k}_reconstruction_metrics.csv").rename(columns={"Reconstruction MSE": "MSE"})

    def run(self):
        rows = []
        for name, k, source in (("FA", self.fa.k, self.fa.metrics), ("ICA", self.ica.k, self.ica.metrics),
                                ("PCA", self.pca_k, self.pca_metrics)):
            table = source[["Dataset", "MSE", "Relative Frobenius"]].copy()
            table["Dataset"] = table["Dataset"].str.capitalize()
            table.insert(0, "Algorithm", name)
            table.insert(1, "Components", k)
            table["RMSE"] = np.sqrt(table["MSE"])
            table["Reconstructed Energy"] = 1-table["Relative Frobenius"]**2
            rows.append(table)
        self.results = pd.concat(rows, ignore_index=True)
        save_table(self.results, NOTEBOOK_RESULTS / "fa_ica_pca_reconstruction_metrics.csv")
        return self
