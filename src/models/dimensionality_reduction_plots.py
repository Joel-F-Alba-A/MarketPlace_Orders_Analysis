"""Plots for notebook 03; model fitting lives in dimensionality_reduction.py."""
import matplotlib.pyplot as plt
import numpy as np


def plot_search(search):
    """Plot reconstruction, retention, stability and clustering diagnostics."""
    retention = ("Retained Energy" if search.method == "svd"
                 else "Cumulative Explained Variance")
    specs = [
        ("Reconstruction MSE", [("Train MSE", "Training"),
                                 ("Validation MSE", "Validation")], "MSE"),
        ("Relative Reconstruction Error", [("Train Relative Frobenius", "Training"),
                                           ("Validation Relative Frobenius", "Validation")],
         "Relative Frobenius error"),
        (retention, [(retention, "Training")], "Retained ratio"),
        ("Subspace Stability", [("Subspace Stability", "Mean stability")], "Mean squared cosine"),
        ("Component Concentration", [("Mean Concentration", "Mean concentration")], "Concentration"),
        ("Clustering Utility", [("Silhouette Score", "Silhouette")], "Silhouette score"),
    ]
    if search.method == "ppca":
        specs.extend([
            ("Stability Standard Deviation", [("Stability SD", "Stability SD")], "Standard deviation"),
            ("Gaussian Log-Likelihood", [("Train Log-Likelihood", "Training"),
                                         ("Validation Log-Likelihood", "Validation")],
             "Average log-likelihood per observation"),
            ("Estimated Noise Variance", [("Estimated Noise Variance", "Noise variance")],
             "Noise variance"),
        ])
    rows = 3 if search.method == "ppca" else 2
    fig, axes = plt.subplots(rows, 3, figsize=(21, 6 * rows), constrained_layout=True)
    frame = search.results.sort_values("k")
    x = frame["k"].to_numpy()
    for ax, (title, series, ylabel) in zip(axes.flat, specs):
        for i, (column, label) in enumerate(series):
            color = ("#1f77b4", "#e3a008")[i]
            y = frame[column].to_numpy(dtype=float)
            if column == "Subspace Stability":
                ax.errorbar(x, y, yerr=frame["Stability SD"], color=color,
                            marker="o", linewidth=2, capsize=4, label="Mean ± 1 SD")
            else:
                ax.plot(x, y, color=color, marker=("o", "s")[i], linewidth=2, label=label)
            for k, value in zip(x, y):
                if np.isfinite(value):
                    ax.annotate(f"{value:.4f}", (k, value), xytext=(0, 12 if i == 0 else -17),
                                textcoords="offset points", ha="center", fontsize=8, color=color)
        ax.set(title=title, xlabel="Number of Components (k)", ylabel=ylabel, xticks=x)
        if title in (retention, "Subspace Stability"):
            ax.set_ylim(0, 1.05)
        elif title == "Clustering Utility":
            ax.set_ylim(-1.05, 1.05)
        elif title != "Gaussian Log-Likelihood":
            ax.set_ylim(bottom=0)
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    name = {"svd": "Truncated SVD", "pca": "Classical PCA", "ppca": "Probabilistic PCA"}[search.method]
    fig.suptitle(f"{name} — Evaluation Metrics Across Components", fontsize=19, fontweight="bold")
    plt.show()


def plot_loadings(analysis):
    """Display the fitted component coefficients (PPCA uses its loading W)."""
    loadings = analysis.loadings
    limit = np.abs(loadings.to_numpy()).max()
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(loadings, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(analysis.k), loadings.columns)
    ax.set_yticks(range(len(loadings)), loadings.index)
    for i in range(len(loadings)):
        for j in range(analysis.k):
            ax.text(j, i, f"{loadings.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"{analysis.method.upper()} — Component Loadings (k={analysis.k})")
    fig.colorbar(image, ax=ax, label="PPCA loading" if analysis.method == "ppca" else "Component coefficient")
    fig.tight_layout()
    plt.show()
