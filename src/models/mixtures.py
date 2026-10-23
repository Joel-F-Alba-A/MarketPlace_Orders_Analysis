"""Reusable GMM experiments, diagnostics, and cluster interpretation.

The notebook supplies configuration and observations; this module owns fitting,
CSV persistence, and model diagnostics. Scaling is fitted on training data only.
"""
from pathlib import Path
from itertools import combinations
import os
import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import CLEAN_DATA_PATH, RESULTS_DIR, SEED

DEFAULT_FEATURES = ("Sales", "Order Item Discount Rate",
                    "Order Item Profit Ratio", "Order Item Quantity")
EXPERIMENT_DIR = RESULTS_DIR / "02_gmm_latent_variables_em"

class GMMData:
    """Clean feature rows and retain their indices for cluster interpretation."""

    def __init__(self, path=CLEAN_DATA_PATH, features=DEFAULT_FEATURES,
                 test_size=0.2, random_state=SEED):
        self.features = list(features)
        self.frame = pd.read_csv(path, encoding="latin-1")
        self.X = self.frame[self.features].replace([np.inf, -np.inf], np.nan).dropna()
        self.train, self.validation = train_test_split(
            self.X, test_size=test_size, random_state=random_state)
        self.scaler = StandardScaler()
        self.train_scaled = self.scaler.fit_transform(self.train)
        self.validation_scaled = self.scaler.transform(self.validation)

class GMMSearch:
    """Cached covariance/component search with incremental CSV reporting."""

    def __init__(self, k_values=range(1, 9),
                 covariance_types=("full", "diag", "spherical", "tied"),
                 output_dir=EXPERIMENT_DIR, n_jobs=-1):
        self.k_values = tuple(k_values)
        self.covariance_types = tuple(covariance_types)
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs

    def run(self, data, force_retrain=False):
        X_train_scaled = data.train_scaled
        X_val_scaled = data.validation_scaled
        EXPERIMENT_DIR = self.output_dir
        MODELS_DIR = EXPERIMENT_DIR / "models"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        N_JOBS = self.n_jobs
        FORCE_RETRAIN = force_retrain
        def train_or_load(k, covariance_type, X_train, X_val):
            cache_path = MODELS_DIR / f"gmm_k{k}_{covariance_type}.joblib"

            if cache_path.exists() and not FORCE_RETRAIN:
                saved = joblib.load(cache_path)
                model = saved["model"]
                metrics = saved["metrics"].copy()

            else:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    init_params="kmeans",
                    n_init=5,
                    max_iter=300,
                    tol=1e-4,
                    reg_covar=1e-5,
                    random_state=SEED,
                )

                model.fit(X_train)

                # Validation metrics evaluate the fitted model on unseen observations.
                val_labels = model.predict(X_val)
                val_probs = model.predict_proba(X_val)
                n_val_clusters = len(np.unique(val_labels))

                if k > 1:
                    entropy = -np.sum(
                        val_probs * np.log(np.clip(val_probs, 1e-12, 1.0)),
                        axis=1,
                    )
                    mean_entropy = float((entropy / np.log(k)).mean())
                else:
                    mean_entropy = np.nan

                silhouette = np.nan
                db = np.nan
                ch = np.nan

                if 1 < n_val_clusters < len(val_labels):
                    try:
                        silhouette = silhouette_score(
                            X_val,
                            val_labels,
                            sample_size=min(5000, len(val_labels)),
                            random_state=SEED,
                        )
                    except ValueError:
                        # A sample may contain too few clusters for silhouette.
                        pass

                    db = davies_bouldin_score(X_val, val_labels)
                    ch = calinski_harabasz_score(X_val, val_labels)

                metrics = {
                    "K": k,
                    "Covariance": covariance_type,
                    "BIC": model.bic(X_train),
                    "AIC": model.aic(X_train),
                    "Train LogLik": model.score(X_train),
                    "Val LogLik": model.score(X_val),
                    "Silhouette": silhouette,
                    "Davies-Bouldin": db,
                    "Calinski-Harabasz": ch,
                    "Entropy": mean_entropy,
                    "Converged": bool(model.converged_),
                    "Iterations": int(model.n_iter_),
                    "Predicted Clusters": n_val_clusters,
                }

            # Count TRAINING observations assigned to each component.
            # This also runs for cached models, without fitting them again.
            train_labels = model.predict(X_train)
            train_counts = np.bincount(train_labels, minlength=k)

            metrics["Train Observations"] = len(train_labels)
            metrics["Train Predicted Clusters"] = int(
                np.count_nonzero(train_counts)
            )

            for cluster_id, count in enumerate(train_counts):
                metrics[f"Train Cluster {cluster_id} Count"] = int(count)
                metrics[f"Train Cluster {cluster_id} %"] = float(
                    100.0 * count / len(train_labels)
                )

            # Save the fitted model and updated metrics atomically.
            temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")

            joblib.dump(
                {"model": model, "metrics": metrics},
                temporary_path,
                compress=3,
            )
            temporary_path.replace(cache_path)

            return metrics


        combinations = [
            (k, covariance_type)
            for covariance_type in self.covariance_types
            for k in self.k_values
        ]

        print(
            f"Evaluating {len(combinations)} combinations "
            f"with {joblib.effective_n_jobs(N_JOBS)} workers."
        )

        results = []

        with parallel_config(
            backend="loky",
            n_jobs=N_JOBS,
            inner_max_num_threads=1,
        ):
            completed = Parallel(
                return_as="generator_unordered",
                max_nbytes="1M",
                mmap_mode="r",
            )(
                delayed(train_or_load)(
                    k,
                    covariance_type,
                    X_train_scaled,
                    X_val_scaled,
                )
                for k, covariance_type in combinations
            )

            # Only the main process writes the shared CSV.
            for metrics in completed:
                results.append(metrics)

                results_df = (
                    pd.DataFrame(results)
                    .sort_values(["Covariance", "K"])
                    .reset_index(drop=True)
                )

                results_df.to_csv(
                    EXPERIMENT_DIR / "gmm_results.csv",
                    index=False,
                )

                counts_text = ", ".join(
                    f"C{cluster_id}: "
                    f"{metrics[f'Train Cluster {cluster_id} Count']:,}"
                    for cluster_id in range(metrics["K"])
                )

                print(
                    f"[{len(results)}/{len(combinations)}] "
                    f"K={metrics['K']}, "
                    f"covariance={metrics['Covariance']}, "
                    f"BIC={metrics['BIC']:.2f} | "
                    f"Training counts: {counts_text}"
                )

        # Full results: metrics, counts, and percentages.
        display(results_df.sort_values("BIC"))

        # Separate table: one row per model and training cluster.
        cluster_rows = []

        for metrics in results:
            for cluster_id in range(metrics["K"]):
                cluster_rows.append({
                    "K": metrics["K"],
                    "Covariance": metrics["Covariance"],
                    "Cluster": cluster_id,
                    "Train Count": metrics[
                        f"Train Cluster {cluster_id} Count"
                    ],
                    "Train %": metrics[
                        f"Train Cluster {cluster_id} %"
                    ],
                })

        cluster_counts_df = (
            pd.DataFrame(cluster_rows)
            .sort_values(["Covariance", "K", "Cluster"])
            .reset_index(drop=True)
        )

        cluster_counts_df.to_csv(
            EXPERIMENT_DIR / "gmm_train_cluster_counts.csv",
            index=False,
        )

        print("\nTraining observations per cluster:")
        display(cluster_counts_df.round({"Train %": 2}))

        # Select the best converged model using BIC.
        converged_results = results_df.loc[results_df["Converged"]]

        if not converged_results.empty:
            best_row = converged_results.loc[
                converged_results["BIC"].idxmin()
            ]

            best_key = (
                int(best_row["K"]),
                best_row["Covariance"],
            )

            best_path = (
                MODELS_DIR / f"gmm_k{best_key[0]}_{best_key[1]}.joblib"
            )
            best_model = joblib.load(best_path)["model"]

            print(
                f"\nBest converged model by BIC: "
                f"K={best_key[0]}, "
                f"covariance={best_key[1]}, "
                f"BIC={best_row['BIC']:.2f}"
            )

            best_cluster_counts = cluster_counts_df.loc[
                (cluster_counts_df["K"] == best_key[0])
                & (cluster_counts_df["Covariance"] == best_key[1])
            ]

            print("\nTraining cluster sizes for the best model:")
            display(best_cluster_counts.round({"Train %": 2}))

        else:
            print(
                "No models converged. "
                "Inspect the fits before selecting a model."
            )
        self.results = results_df
        self.cluster_counts = cluster_counts_df
        self.best_model = best_model if not converged_results.empty else None
        return self

class GMMStability:
    """Compare seeds and initialization counts; reuse matching cached results."""

    def __init__(self, k=4, covariance_type="diag",
                 seeds=(0, 10, 21, 42, 100, 123, 202, 500, 1000, 2026),
                 n_init_values=(1, 10), n_jobs=-1, output_dir=None):
        self.k = k
        self.covariance_type = covariance_type
        self.seeds = tuple(seeds)
        self.n_init_values = tuple(n_init_values)
        self.n_jobs = n_jobs
        self.output_dir = Path(output_dir) if output_dir is not None else EXPERIMENT_DIR / "seed_stability"

    def load_saved(self):
        """Load exported CSVs without fitting; use run(data) for a new experiment."""
        self.runs = pd.read_csv(self.output_dir / "runs.csv")
        self.pairwise_ari = pd.read_csv(self.output_dir / "pairwise_ari.csv")
        self.summary = pd.read_csv(self.output_dir / "comparison_summary.csv", index_col="n_init")
        self.labels = pd.read_csv(self.output_dir / "validation_labels.csv", index_col="Validation_Row")
        expected = {(n, seed) for n in self.n_init_values for seed in self.seeds}
        actual = set(self.runs[["n_init", "Seed"]].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError("Saved seeds/n_init differ from this configuration. Use run(data).")
        self.ari_matrices = {}
        for n_init in self.n_init_values:
            matrix = pd.read_csv(self.output_dir / f"ari_matrix_n_init_{n_init}.csv", index_col="Seed")
            matrix.columns = matrix.columns.astype(int)
            self.ari_matrices[n_init] = matrix
        return self

    def run(self, data, force_retrain=False):
        seeds, n_init_values, N_JOBS = self.seeds, self.n_init_values, self.n_jobs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = joblib.hash((data.train_scaled, data.validation_scaled,
                                   self.k, self.covariance_type, seeds, n_init_values))
        cache_path = self.output_dir / "stability.joblib"
        if cache_path.exists() and not force_retrain:
            saved = joblib.load(cache_path)
            if saved["fingerprint"] == fingerprint:
                for name in ("runs", "pairwise_ari", "ari_matrices", "summary", "labels"):
                    setattr(self, name, saved[name])
                return self
        def fit_one_model(seed, n_init):

            with threadpool_limits(limits=1):

                model = GaussianMixture(
                    n_components=self.k,
                    covariance_type=self.covariance_type,
                    init_params="kmeans",
                    n_init=n_init,
                    max_iter=300,
                    tol=1e-4,
                    reg_covar=1e-5,
                    random_state=seed
                )

                model.fit(data.train_scaled)

                labels = model.predict(data.validation_scaled)

                metrics = {
                    "n_init": n_init,
                    "Seed": seed,
                    "Converged": model.converged_,
                    "Iterations": model.n_iter_,
                    "Train LogLik": model.score(data.train_scaled),
                    "Val LogLik": model.score(data.validation_scaled),
                    "BIC": model.bic(data.train_scaled),
                    "Predicted Clusters": len(np.unique(labels))
                }

                return n_init, seed, labels, metrics


        # 20 models: 10 seeds × 2 n_init values
        outputs = Parallel(
            n_jobs=N_JOBS,
            verbose=10
        )(
            delayed(fit_one_model)(seed, n_init)
            for n_init in n_init_values
            for seed in seeds
        )

        # ==========================================
        # 3. RESULTS BY SEED AND N_INIT
        # ==========================================

        runs_df = pd.DataFrame([
            metrics
            for n_init, seed, labels, metrics in outputs
        ])

        labels_by_config = {
            (n_init, seed): labels
            for n_init, seed, labels, metrics in outputs
        }

        print("\nRESULTS BY CONFIGURATION")
        print(
            runs_df
            .sort_values(["n_init", "Seed"])
            .round(4)
            .to_string(index=False)
        )

        # ==========================================
        # 4. PAIRWISE ARI FOR EACH N_INIT
        # ==========================================

        ari_results = []
        ari_matrices = {}

        for n_init in n_init_values:

            ari_matrix = pd.DataFrame(
                np.eye(len(seeds)),
                index=seeds,
                columns=seeds
            )

            for seed_a, seed_b in combinations(seeds, 2):

                ari = adjusted_rand_score(
                    labels_by_config[(n_init, seed_a)],
                    labels_by_config[(n_init, seed_b)]
                )

                ari_results.append({
                    "n_init": n_init,
                    "Seed A": seed_a,
                    "Seed B": seed_b,
                    "ARI": ari
                })

                ari_matrix.loc[seed_a, seed_b] = ari
                ari_matrix.loc[seed_b, seed_a] = ari

            ari_matrices[n_init] = ari_matrix

        ari_df = pd.DataFrame(ari_results)

        # ==========================================
        # 5. STABILITY SUMMARY
        # ==========================================

        stability_summary = (
            ari_df.groupby("n_init")["ARI"]
            .agg(
                Mean_ARI="mean",
                Median_ARI="median",
                Min_ARI="min",
                Max_ARI="max",
                Std_ARI="std"
            )
        )

        fit_summary = (
            runs_df.groupby("n_init")
            .agg(
                Mean_Train_LogLik=("Train LogLik", "mean"),
                Std_Train_LogLik=("Train LogLik", "std"),
                Mean_Val_LogLik=("Val LogLik", "mean"),
                Std_Val_LogLik=("Val LogLik", "std"),
                Mean_BIC=("BIC", "mean"),
                Mean_Iterations=("Iterations", "mean"),
                Convergence_Rate=("Converged", "mean")
            )
        )

        comparison_summary = stability_summary.join(fit_summary)
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results for each configuration
        runs_df.sort_values(["n_init", "Seed"]).to_csv(
            output_dir / "runs.csv",
            index=False
        )

        # Save pairwise ARI scores
        ari_df.to_csv(
            output_dir / "pairwise_ari.csv",
            index=False
        )

        # Save stability and fit summaries
        comparison_summary.to_csv(
            output_dir / "comparison_summary.csv",
            index_label="n_init"
        )

        # Save ARI matrices
        for n_init, matrix in ari_matrices.items():
            matrix.to_csv(
                output_dir / f"ari_matrix_n_init_{n_init}.csv",
                index_label="Seed"
            )

        # Save validation cluster assignments
        labels_df = pd.DataFrame({
            f"n_init_{n_init}_seed_{seed}": labels
            for (n_init, seed), labels in labels_by_config.items()
        })

        labels_df.to_csv(
            output_dir / "validation_labels.csv",
            index_label="Validation_Row"
        )

        print(f"Results saved to: {output_dir}")
        self.runs = runs_df
        self.pairwise_ari = ari_df
        self.ari_matrices = ari_matrices
        self.summary = comparison_summary
        self.labels = labels_df
        joblib.dump({"fingerprint": fingerprint, "runs": self.runs,
                     "pairwise_ari": self.pairwise_ari, "ari_matrices": self.ari_matrices,
                     "summary": self.summary, "labels": self.labels}, cache_path)
        return self

class GMMAnalysis:
    """Selected model, EM diagnostics, and original-unit cluster profiles."""

    def __init__(self, data, k=4, covariance_type="diag", n_init=10,
                 random_state=SEED):
        self.data = data
        self.model = GaussianMixture(
            n_components=k, covariance_type=covariance_type, n_init=n_init,
            max_iter=300, tol=1e-4, reg_covar=1e-5, random_state=random_state)

    def fit(self):
        self.model.fit(self.data.train_scaled)
        self.cluster_data = self.data.frame.loc[self.data.validation.index].copy()
        self.cluster_data["Cluster"] = self.model.predict(self.data.validation_scaled)
        return self

    def convergence_summary(self):
        return pd.Series({
            "Converged": self.model.converged_,
            "EM iterations": self.model.n_iter_,
            "Final lower bound": self.model.lower_bound_,
            "Training LogLik": self.model.score(self.data.train_scaled),
            "Validation LogLik": self.model.score(self.data.validation_scaled)})

    def responsibilities_summary(self):
        # E-step: Posterior probabilities
        responsibilities = self.model.predict_proba(self.data.validation_scaled)

        # Maximum membership probability for each observation
        max_probabilities = responsibilities.max(axis=1)

        # Normalized entropy (0 = certain, 1 = maximum uncertainty)
        entropy = -np.sum(
            responsibilities * np.log(responsibilities + 1e-12),
            axis=1
        ) / np.log(responsibilities.shape[1])

        # Summary statistics
        summary = pd.Series({
            "Number of observations": len(responsibilities),
            "Mean max probability": max_probabilities.mean(),
            "Median max probability": np.median(max_probabilities),
            "Percentage with max probability < 0.6":
                100 * np.mean(max_probabilities < 0.6),
            "Percentage with max probability >= 0.9":
                100 * np.mean(max_probabilities >= 0.9),
            "Mean normalized entropy": entropy.mean()
        })

        return summary

    def component_parameters(self):
        """Return weights, original-unit means, and standardized variances."""
        model = self.model
        if model.covariance_type == "diag":
            variances = model.covariances_
        elif model.covariance_type == "full":
            variances = np.diagonal(model.covariances_, axis1=1, axis2=2)
        elif model.covariance_type == "tied":
            variances = np.tile(np.diag(model.covariances_), (model.n_components, 1))
        else:
            variances = np.repeat(model.covariances_[:, None], len(self.data.features), axis=1)
        tables = {
            "Mixture weights": pd.DataFrame({"Component": range(model.n_components), "Weight": model.weights_}),
            "Component means (original units)": pd.DataFrame(
                self.data.scaler.inverse_transform(model.means_), columns=self.data.features),
            "Component variances (standardized units)": pd.DataFrame(variances, columns=self.data.features)}
        for title, table in tables.items():
            table.index.name = "Component"
            print(title)
            display(table)
        return tables

    def analyze_cluster(self, cluster_id):
        data = self.cluster_data
        features = self.data.features
        if cluster_id not in data["Cluster"].unique():
            raise ValueError(f"Cluster {cluster_id} does not exist.")

        # Filter observations belonging to the selected cluster
        cluster = data[data["Cluster"] == cluster_id]

        # Cluster size
        count = len(cluster)
        percentage = 100 * count / len(data)

        print(f"\n{'=' * 45}")
        print(f"CLUSTER {cluster_id} ANALYSIS")
        print(f"{'=' * 45}")

        print(f"Number of observations: {count:,}")
        print(f"Percentage of validation data: {percentage:.2f}%")

        # Descriptive statistics
        print("\nDESCRIPTIVE STATISTICS")

        statistics = cluster[features].describe().T[
            ["mean", "std", "min", "25%", "50%", "75%", "max"]
        ].rename(columns={
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3"
        })

        display(statistics.round(4))

        # Profitability analysis
        negative_profit = (
            cluster["Order Item Profit Ratio"] < 0
        ).mean() * 100

        print("\nPROFITABILITY ANALYSIS")
        print(f"Negative profit observations: {negative_profit:.2f}%")

        return cluster[features]

    def category_counts(self, cluster_id, column, top=None):
        counts = self.cluster_data.loc[self.cluster_data["Cluster"] == cluster_id, column].value_counts()
        return counts if top is None else counts.head(top)

    def plot_cluster(self, cluster_id):
        from src.models.gmm_plots import plot_cluster
        return plot_cluster(cluster_id, self.cluster_data, self.data.features)
