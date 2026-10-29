"""Reusable data, searches and analyses for notebook 03.

The notebook owns model choices and interpretation. This module preserves its
70/15/15 split, training-only scaling, bootstrap diagnostics and PPCA posterior
reconstruction. PPCA search diagnostics use PCA scores/reconstruction, as in the
original notebook; the final PPCA analysis uses posterior latent means.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN_DATA_PATH, SEED

DEFAULT_FEATURES = (
    "Days for shipping (real)", "Days for shipment (scheduled)",
    "Benefit per order", "Sales per customer", "Order Item Discount",
    "Order Item Discount Rate", "Order Item Profit Ratio", "Order Item Quantity",
    "Sales", "Product Price",
)


class ReductionData:
    """Keep feature names/indices and fit both scalers on training data only."""

    def __init__(self, path=CLEAN_DATA_PATH, features=DEFAULT_FEATURES,
                 random_state=SEED, frame=None):
        self.features = list(features)
        self.random_state = random_state
        self.frame = pd.read_csv(Path(path)) if frame is None else frame.copy()
        self.X = self.frame.loc[:, self.features].copy()
        if not np.isfinite(self.X.to_numpy(dtype=float)).all():
            raise ValueError("Reduction features must contain only finite numeric values.")
        self.train, temporary = train_test_split(
            self.X, test_size=0.30, random_state=random_state)
        self.validation, self.test = train_test_split(
            temporary, test_size=0.50, random_state=random_state)
        self.svd_scaler = StandardScaler(with_mean=False)
        self.scaler = StandardScaler()
        self.train_scaled = self.svd_scaler.fit_transform(self.train)
        self.validation_scaled = self.svd_scaler.transform(self.validation)
        self.test_scaled = self.svd_scaler.transform(self.test)
        self.train_standardized = self.scaler.fit_transform(self.train)
        self.validation_standardized = self.scaler.transform(self.validation)
        self.test_standardized = self.scaler.transform(self.test)

    def arrays(self, method):
        """Return train/validation/test arrays in the method's original scaling."""
        _check_method(method)
        suffix = "scaled" if method == "svd" else "standardized"
        return tuple(getattr(self, f"{split}_{suffix}")
                     for split in ("train", "validation", "test"))


def _check_method(method):
    if method not in ("svd", "pca", "ppca"):
        raise ValueError("method must be 'svd', 'pca', or 'ppca'.")


def _estimator(method, k, random_state):
    if method == "svd":
        return TruncatedSVD(n_components=k, algorithm="arpack",
                            random_state=random_state)
    return PCA(n_components=k, svd_solver="full")


def relative_frobenius_error(X, reconstructed):
    denominator = np.linalg.norm(X, ord="fro")
    return (np.linalg.norm(X - reconstructed, ord="fro") / denominator
            if denominator else np.nan)


def component_concentration(components):
    """Sum fourth powers of each unit-length component's coefficients."""
    return np.sum(components ** 4, axis=1)


def clustering_utility(Z, n_clusters=3, sample_size=5000, random_state=SEED):
    """Preserve the search's sampled, standardized-latent clustering diagnostic."""
    if Z.shape[1] < 2 or len(Z) <= n_clusters:
        return np.nan
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(Z), size=min(sample_size, len(Z)), replace=False)
    sample = StandardScaler().fit_transform(Z[indices])
    if len(sample) <= n_clusters:
        return np.nan
    labels = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=random_state).fit_predict(sample)
    if not 1 < len(np.unique(labels)) < len(sample):
        return np.nan
    return silhouette_score(sample, labels, sample_size=min(2000, len(sample)),
                            random_state=random_state)


class ReductionSearch:
    """Evaluate component counts using the metrics from the original notebook."""

    def __init__(self, method, k_values=None, n_bootstrap=10, n_clusters=3,
                 sample_size=5000, random_state=SEED):
        _check_method(method)
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1.")
        self.method = method
        self.k_values = None if k_values is None else tuple(k_values)
        self.n_bootstrap = n_bootstrap
        self.n_clusters = n_clusters
        self.sample_size = sample_size
        self.random_state = random_state

    def _stability(self, X, k, components):
        rng = np.random.default_rng(self.random_state)
        similarities = []
        for _ in range(self.n_bootstrap):
            indices = rng.choice(len(X), size=len(X), replace=True)
            boot = _estimator(self.method, k, self.random_state).fit(X[indices])
            angles = subspace_angles(components.T, boot.components_.T)
            similarities.append(np.mean(np.cos(angles) ** 2))
        return float(np.mean(similarities)), float(np.std(similarities))

    def run(self, data):
        train, validation, _ = data.arrays(self.method)
        max_k = min(train.shape) - 1
        counts = tuple(range(1, max_k + 1)) if self.k_values is None else self.k_values
        if not counts or any(not isinstance(k, (int, np.integer)) or not 1 <= k <= max_k
                             for k in counts):
            raise ValueError(f"k_values must contain integers between 1 and {max_k}.")
        rows = []
        self.models = {}
        self.loadings_by_k = {}
        for k in counts:
            model = _estimator(self.method, k, self.random_state)
            latent = model.fit_transform(train)
            train_rec = model.inverse_transform(latent)
            val_rec = model.inverse_transform(model.transform(validation))
            stability, stability_sd = self._stability(train, k, model.components_)
            metrics = {
                "k": k,
                "Train MSE": mean_squared_error(train, train_rec),
                "Validation MSE": mean_squared_error(validation, val_rec),
                "Train Relative Frobenius": relative_frobenius_error(train, train_rec),
                "Validation Relative Frobenius": relative_frobenius_error(validation, val_rec),
            }
            if self.method == "svd":
                metrics["Retained Energy"] = (np.sum(model.singular_values_ ** 2)
                                               / np.sum(train ** 2))
            else:
                metrics["Cumulative Explained Variance"] = model.explained_variance_ratio_.sum()
            metrics.update({
                "Subspace Stability": stability,
                "Stability SD": stability_sd,
                "Mean Concentration": component_concentration(model.components_).mean(),
                "Silhouette Score": clustering_utility(
                    latent, self.n_clusters, self.sample_size, self.random_state),
            })
            if self.method == "ppca":
                metrics.update({
                    "Train Log-Likelihood": model.score(train),
                    "Validation Log-Likelihood": model.score(validation),
                    "Estimated Noise Variance": model.noise_variance_,
                })
            prefix = "SVD" if self.method == "svd" else "PC"
            self.loadings_by_k[k] = pd.DataFrame(
                model.components_.T, index=data.features,
                columns=[f"{prefix}_{i + 1}" for i in range(k)])
            self.models[k] = model
            rows.append(metrics)
        self.results = pd.DataFrame(rows)
        return self


class SVDSearch(ReductionSearch):
    def __init__(self, **kwargs):
        super().__init__("svd", **kwargs)


class PCASearch(ReductionSearch):
    def __init__(self, **kwargs):
        super().__init__("pca", **kwargs)


class PPCASearch(ReductionSearch):
    def __init__(self, **kwargs):
        super().__init__("ppca", **kwargs)


class ReductionAnalysis:
    """Fit the selected model and expose loadings, scores and reconstructions."""

    def __init__(self, data, method, k=5, random_state=SEED):
        _check_method(method)
        self.data = data
        self.method = method
        self.k = k
        self.random_state = random_state

    def fit(self):
        arrays = self.data.arrays(self.method)
        if not isinstance(self.k, (int, np.integer)) or not 1 <= self.k < min(arrays[0].shape):
            raise ValueError("k must be positive and smaller than both training dimensions.")
        self.model = _estimator(self.method, self.k, self.random_state).fit(arrays[0])
        if self.method == "ppca":
            # Rotation R=I, preserving the notebook's sklearn covariance convention.
            self.W = self.model.components_.T @ np.diag(np.sqrt(np.maximum(
                self.model.explained_variance_ - self.model.noise_variance_, 0)))
        self.latent = {}
        self.reconstructions = {}
        for name, X in zip(("train", "validation", "test"), arrays):
            Z = self.transform(X)
            self.latent[name] = Z
            self.reconstructions[name] = self.inverse_transform(Z)
        prefix = {"svd": "SVD_", "pca": "PC", "ppca": "Latent "}[self.method]
        self.component_names = [f"{prefix}{i + 1}" for i in range(self.k)]
        self.loadings = pd.DataFrame(
            self.W if self.method == "ppca" else self.model.components_.T,
            index=self.data.features, columns=self.component_names)
        return self

    def transform(self, X):
        """Project arrays already scaled with this analysis's training scaler."""
        if self.method != "ppca":
            return self.model.transform(X)
        centered = np.asarray(X) - self.model.mean_
        M = self.W.T @ self.W + self.model.noise_variance_ * np.eye(self.k)
        return np.linalg.solve(M, self.W.T @ centered.T).T

    def inverse_transform(self, Z):
        if self.method == "ppca":
            return np.asarray(Z) @ self.W.T + self.model.mean_
        return self.model.inverse_transform(Z)

    def latent_frame(self, split="train"):
        return pd.DataFrame(self.latent[split], columns=self.component_names,
                            index=getattr(self.data, split).index)

    def variance_summary(self):
        if self.method == "svd":
            raise ValueError("SVD reports retained energy in its search results.")
        ratio = self.model.explained_variance_ratio_
        return pd.DataFrame({"Component": self.component_names,
                             "Explained Variance Ratio": ratio,
                             "Cumulative Explained Variance": np.cumsum(ratio)})

    def evaluate(self, split="test"):
        X = self.data.arrays(self.method)[("train", "validation", "test").index(split)]
        result = {
            "Reconstruction MSE": mean_squared_error(X, self.reconstructions[split]),
            "Relative Frobenius": relative_frobenius_error(X, self.reconstructions[split]),
        }
        if self.method == "ppca":
            result["Average Log-Likelihood"] = self.model.score(X)
        return result

    def metrics(self):
        return pd.DataFrame({s: self.evaluate(s)
                             for s in ("train", "validation", "test")}).T


class SVDAnalysis(ReductionAnalysis):
    def __init__(self, data, k=5, random_state=SEED):
        super().__init__(data, "svd", k, random_state)


class PCAAnalysis(ReductionAnalysis):
    def __init__(self, data, k=5, random_state=SEED):
        super().__init__(data, "pca", k, random_state)


class PPCAAnalysis(ReductionAnalysis):
    def __init__(self, data, k=5, random_state=SEED):
        super().__init__(data, "ppca", k, random_state)


class ReductionComparison:
    """Compare selected models on test data in their respective scaled spaces."""

    def __init__(self, *analyses, n_clusters=4, random_state=SEED):
        self.analyses = analyses
        self.n_clusters = n_clusters
        self.random_state = random_state

    def run(self):
        rows = []
        names = {"svd": "Truncated SVD", "pca": "Classical PCA", "ppca": "PPCA"}
        for analysis in self.analyses:
            X = analysis.data.arrays(analysis.method)[2]
            Z = analysis.latent["test"]
            reconstructed = analysis.reconstructions["test"]
            # The original final comparison fits K-means on all test scores,
            # without the search diagnostic's extra latent standardization.
            silhouette = np.nan
            if len(Z) > self.n_clusters:
                labels = KMeans(n_clusters=self.n_clusters, n_init=10,
                                random_state=self.random_state).fit_predict(Z)
                if 1 < len(np.unique(labels)) < len(Z):
                    silhouette = silhouette_score(
                        Z, labels, sample_size=min(5000, len(Z)),
                        random_state=self.random_state)
            mse = mean_squared_error(X, reconstructed)
            is_svd = analysis.method == "svd"
            rows.append({
                "Model": names[analysis.method], "Components": Z.shape[1],
                "Test MSE": mse, "Test RMSE": np.sqrt(mse),
                "Relative Frobenius": relative_frobenius_error(X, reconstructed),
                "Retained Energy": (np.linalg.norm(reconstructed, ord="fro") ** 2
                                    / np.linalg.norm(X, ord="fro") ** 2) if is_svd else np.nan,
                "Explained Variance": (np.nan if is_svd else
                                       analysis.model.explained_variance_ratio_.sum()),
                "Silhouette": silhouette,
                "Avg Log-Likelihood": (analysis.model.score(X)
                                       if analysis.method == "ppca" else np.nan),
            })
        self.results = pd.DataFrame(rows)
        return self
