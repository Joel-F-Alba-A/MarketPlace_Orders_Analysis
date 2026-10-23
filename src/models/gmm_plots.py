"""Plotting helpers for the GMM notebook; fitting lives in mixtures.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_search(search):
    results_df = search.results
    output_dir = search.output_dir
    plot_specs = [
        ("BIC", "BIC ↓"),
        ("AIC", "AIC ↓"),
        ("Train LogLik", "Training mean log-likelihood ↑"),
        ("Val LogLik", "Validation mean log-likelihood ↑"),
        ("Silhouette", "Silhouette ↑"),
        ("Davies-Bouldin", "Davies–Bouldin ↓"),
        ("Calinski-Harabasz", "Calinski–Harabasz ↑"),
        ("Entropy", "Normalized assignment entropy"),
        ("Iterations", "EM iterations"),
    ]

    covariance_order = ["full", "diag", "spherical", "tied"]
    colors = dict(zip(covariance_order, plt.get_cmap("tab10").colors[:4]))

    # Only these metrics should always start at zero.
    zero_based_metrics = {
        "Davies-Bouldin",
        "Calinski-Harabasz",
        "Entropy",
        "Iterations",
    }

    converged = (
        results_df["Converged"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    fig, axes = plt.subplots(
        3, 3,
        figsize=(18, 13),
        sharex=True,
        constrained_layout=True,
    )

    for ax, (metric, title) in zip(axes.flat, plot_specs):
        for covariance_type in covariance_order:
            subset = (
                results_df.loc[
                    results_df["Covariance"] == covariance_type
                ]
                .sort_values("K")
            )

            if subset.empty:
                continue

            ax.plot(
                subset["K"],
                subset[metric],
                marker="o",
                color=colors[covariance_type],
                label=covariance_type,
            )

            failed = subset.loc[~converged.loc[subset.index]]

            ax.scatter(
                failed["K"],
                failed[metric],
                marker="x",
                s=110,
                linewidths=2,
                color="black",
                zorder=5,
            )

        ax.set_title(title)
        ax.set_xlabel("Number of components (K)")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(results_df["K"].unique()))

        values = pd.to_numeric(results_df[metric], errors="coerce").dropna()

        if not values.empty and values.ge(0).all():
            ax.set_ylim(bottom=0)
        else:
            ax.margins(y=0.1)

        # Show actual values instead of an axis offset.
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        title="Covariance type",
    )

    fig.suptitle(
        "GMM comparison — black X marks a non-converged fit",
        fontsize=16,
    )

    figure_path = output_dir / "gmm_metrics_comparison.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Plot saved to: {figure_path}")

def plot_stability(stability):
    runs_df = stability.runs
    ari_matrices = stability.ari_matrices
    seeds = stability.seeds
    n_init_values = stability.n_init_values
    fig, axes = plt.subplots(
        1, len(n_init_values),
        squeeze=False,
        figsize=(7.5 * len(n_init_values), 6),
        constrained_layout=True
    )

    axes = axes.ravel()

    for ax, n_init in zip(axes, n_init_values):

        matrix = ari_matrices[n_init]

        im = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=-1,
            vmax=1
        )

        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(seeds, rotation=45)

        ax.set_yticks(range(len(seeds)))
        ax.set_yticklabels(seeds)

        ax.set_xlabel("Seed")
        ax.set_ylabel("Seed")

        ax.set_title(
            f"ARI Stability — n_init={n_init}"
        )

        for i in range(len(seeds)):
            for j in range(len(seeds)):

                value = matrix.iloc[i, j]

                ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.5 else "black"
                )

    fig.colorbar(
        im,
        ax=axes,
        label="Adjusted Rand Index",
        shrink=0.8
    )

    plt.show()

    # ==========================================
    # 7. LOG-LIKELIHOOD COMPARISON
    # ==========================================

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    for n_init in n_init_values:

        subset = (
            runs_df[runs_df["n_init"] == n_init]
            .set_index("Seed")
            .loc[list(seeds)]
        )

        axes[0].plot(
            range(len(seeds)),
            subset["Train LogLik"],
            marker="o",
            label=f"n_init={n_init}"
        )

        axes[1].plot(
            range(len(seeds)),
            subset["Val LogLik"],
            marker="o",
            label=f"n_init={n_init}"
        )

    axes[0].set_title("Training Log-Likelihood")
    axes[1].set_title("Validation Log-Likelihood")

    for ax in axes:

        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(seeds, rotation=45)

        ax.set_xlabel("Seed")
        ax.set_ylabel("Average Log-Likelihood")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 8. EM ITERATIONS COMPARISON
    # ==========================================

    iterations_table = runs_df.pivot(
        index="Seed",
        columns="n_init",
        values="Iterations"
    ).loc[list(seeds)]

    iterations_table.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("EM Iterations by Seed and n_init")
    plt.xlabel("Seed")
    plt.ylabel("Iterations")
    plt.legend(title="n_init")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_cluster(cluster_id, data, features):

    if cluster_id not in data["Cluster"].unique():
        raise ValueError(f"Cluster {cluster_id} does not exist.")

    cluster = data[data["Cluster"] == cluster_id]

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        nrows=len(features),
        ncols=2,
        figsize=(14, 4 * len(features)),
        squeeze=False
    )

    for i, feature in enumerate(features):

        values = cluster[feature].dropna()

        # ==================================
        # HISTOGRAM
        # ==================================
        sns.histplot(
            values,
            bins=30,
            kde=values.nunique() > 10,
            ax=axes[i, 0],
            color="steelblue"
        )

        axes[i, 0].axvline(
            values.mean(),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Mean"
        )

        axes[i, 0].axvline(
            values.median(),
            color="green",
            linestyle=":",
            linewidth=1.5,
            label="Median"
        )

        axes[i, 0].set_title(f"{feature} - Histogram")
        axes[i, 0].set_xlabel(feature)
        axes[i, 0].set_ylabel("Frequency")
        axes[i, 0].legend()

        # ==================================
        # BOXPLOT
        # ==================================
        sns.boxplot(
            x=values,
            ax=axes[i, 1],
            color="lightblue",
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "red",
                "markeredgecolor": "black",
                "markersize": 7
            }
        )

        axes[i, 1].set_title(
            f"{feature} - Boxplot (Red diamond = Mean)"
        )
        axes[i, 1].set_xlabel(feature)

    fig.suptitle(
        f"Cluster {cluster_id} - Numerical Distributions (n={len(cluster):,})",
        fontsize=16,
        fontweight="bold",
        y=1.01
    )

    plt.tight_layout()
    plt.show()
