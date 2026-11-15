"""Notebook 06 dashboards and anomaly-score presentation helpers."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from IPython.display import display

def plot_pca_distributions(detector):
    val_results = detector.results['validation']
    ALPHAS = detector.alphas
    thresholds = detector.thresholds
    output_dir = detector.results_dir
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = [('Hotelling_T2', 'T2', "Hotelling's T²"), ('SPE', 'SPE', 'Squared Prediction Error (SPE)')]
    for ax, (column, key, title) in zip(axes, metrics):
        values = val_results[column].to_numpy()
        upper = np.quantile(values, 0.999)
        ax.hist(values, bins=60, alpha=0.75, edgecolor='white')
        for alpha in ALPHAS:
            threshold = thresholds[alpha][key]
            ax.axvline(threshold, linestyle='--', linewidth=2, label=f'α = {alpha} ({(1 - alpha) * 100:.0f}th percentile)')
        max_threshold = max((thresholds[alpha][key] for alpha in ALPHAS))
        ax.set_xlim(0, max(upper, max_threshold * 1.1, 1e-08))
        ax.set_title(title)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.2)
    plt.suptitle('PCA Anomaly Detection — Quantile Thresholds (5 Components)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_anomaly_quantile_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_lof_search(experiment):
    score_store = experiment.score_store
    lof_tuning_metrics = experiment.lof_tuning_metrics
    lof_sensitivity_df = experiment.lof_sensitivity_df
    spearman_matrix = experiment.spearman_matrix
    jaccard_matrix = experiment.jaccard_matrix
    output_dir = experiment.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    K_VALUES = sorted(score_store.keys())
    metrics = lof_tuning_metrics.sort_values('K')
    sensitivity = lof_sensitivity_df.copy()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(19, 11), constrained_layout=True)
    ax = axes[0, 0]
    ax.boxplot([score_store[k] for k in K_VALUES], tick_labels=[str(k) for k in K_VALUES], showmeans=True, showfliers=False, meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 6})
    ax.set_title('LOF Score Distribution')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('LOF Anomaly Score')
    ax.grid(axis='y', alpha=0.3)
    ax = axes[0, 1]
    plot_limit = max((np.quantile(score_store[k], 0.99) for k in K_VALUES))
    plot_min = min((np.min(score_store[k]) for k in K_VALUES))
    bins = np.linspace(plot_min, plot_limit, 60)
    for k in K_VALUES:
        scores = score_store[k]
        visible_scores = scores[scores <= plot_limit]
        ax.hist(visible_scores, bins=bins, density=True, alpha=0.35, label=f'k = {k}')
    ax.set_title('LOF Score Histogram (Central Range)')
    ax.set_xlabel('LOF Anomaly Score')
    ax.set_ylabel('Density')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax = axes[0, 2]
    ax.plot(metrics['K'], metrics['Score Std'], marker='o', linewidth=2)
    ax.set_title('Score Dispersion')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Score Standard Deviation')
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(metrics['K'], metrics['Score P95'], marker='o', linewidth=2, label='95th percentile')
    ax.plot(metrics['K'], metrics['Score P99'], marker='s', linewidth=2, label='99th percentile')
    ax.set_title('Upper Score Quantiles')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('LOF Anomaly Score')
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    ax.plot(metrics['K'], metrics['Execution Time (s)'], marker='o', linewidth=2)
    ax.set_title('Execution Time')
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)
    ax = axes[1, 2]
    ax.plot(sensitivity['K Pair'], sensitivity['Spearman'], marker='o', linewidth=2, label='Spearman')
    ax.plot(sensitivity['K Pair'], sensitivity['Top 1% Jaccard'], marker='s', linewidth=2, label='Top-1% Jaccard')
    ax.set_title('Neighborhood Sensitivity')
    ax.set_xlabel('Compared Neighborhood Sizes')
    ax.set_ylabel('Similarity')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle('Local Outlier Factor - Hyperparameter Evaluation', fontsize=17, fontweight='bold')
    plt.savefig(output_dir / 'lof_tuning_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(spearman_matrix.to_numpy(), vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(K_VALUES)), K_VALUES)
    ax.set_yticks(range(len(K_VALUES)), K_VALUES)
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Number of Neighbors (k)')
    ax.set_title('LOF - Spearman Ranking Correlation')
    for i in range(len(K_VALUES)):
        for j in range(len(K_VALUES)):
            value = spearman_matrix.iloc[i, j]
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, label='Spearman Correlation')
    plt.tight_layout()
    plt.savefig(output_dir / 'lof_tuning_spearman_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(jaccard_matrix.to_numpy(), vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(K_VALUES)), K_VALUES)
    ax.set_yticks(range(len(K_VALUES)), K_VALUES)
    ax.set_xlabel('Number of Neighbors (k)')
    ax.set_ylabel('Number of Neighbors (k)')
    ax.set_title('LOF - Top-1% Jaccard Similarity')
    for i in range(len(K_VALUES)):
        for j in range(len(K_VALUES)):
            value = jaccard_matrix.iloc[i, j]
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white' if value < 0.65 else 'black')
    fig.colorbar(im, ax=ax, label='Jaccard Similarity')
    plt.tight_layout()
    plt.savefig(output_dir / 'lof_tuning_jaccard_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('\nAll LOF evaluation figures saved successfully.')


def plot_lof_stability(experiment):
    spearman_seed_matrix = experiment.spearman_seed_matrix
    jaccard_seed_matrix = experiment.jaccard_seed_matrix
    output_dir = experiment.results_dir
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    heatmaps = [(spearman_seed_matrix, 'Spearman Ranking Correlation', 'coolwarm', -1, 1), (jaccard_seed_matrix, 'Top-1% Jaccard Similarity', 'viridis', 0, 1)]
    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, heatmaps):
        values = matrix.to_numpy(dtype=float)
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Seed')
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8, color='white' if value < 0.65 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'LOF (k={experiment.n_neighbors}) - Training-Subsample Stability', fontsize=16, fontweight='bold')
    plt.savefig(output_dir / f'lof_k{experiment.n_neighbors}_seed_stability_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_if_search(experiment):
    if_tuning_metrics = experiment.if_tuning_metrics
    score_store_if = experiment.score_store_if
    output_dir = experiment.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = if_tuning_metrics.copy()
    sample_sizes = sorted(metrics_df['Max Samples'].unique(), key=lambda x: float('inf') if float(x) == 1.0 else float(x))
    tree_counts = sorted(metrics_df['N Estimators'].unique())

    def sample_label(value):
        if float(value) == 1.0:
            return 'All Data'
        return str(int(value))

    def config_key(n_trees, n_samples):
        sample_value = '1.0' if float(n_samples) == 1.0 else str(int(n_samples))
        return f'Trees_{int(n_trees)}_Samples_{sample_value}'
    metrics_to_plot = [('Score Mean', 'Mean Anomaly Score'), ('Score Median', 'Median Anomaly Score'), ('Score Std', 'Score Standard Deviation'), ('Score P95', '95th Percentile Score'), ('Score P99', '99th Percentile Score'), ('Execution Time (s)', 'Execution Time')]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics_to_plot):
        for max_samples in sample_sizes:
            subset = metrics_df[metrics_df['Max Samples'] == max_samples].sort_values('N Estimators')
            ax.plot(subset['N Estimators'], subset[metric], marker='o', linewidth=2, label=sample_label(max_samples))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Number of Trees')
        ax.set_ylabel(metric)
        ax.set_xticks(tree_counts)
        ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=min(len(sample_sizes), 5), frameon=False)
    fig.suptitle('Isolation Forest - Hyperparameter Evaluation', fontsize=16, fontweight='bold')
    fig.savefig(output_dir / 'if_hyperparameter_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    configurations = [(n_trees, n_samples) for n_samples in sample_sizes for n_trees in tree_counts]
    config_names = [config_key(n_trees, n_samples) for n_trees, n_samples in configurations]
    config_labels = [f'{int(n_trees)}T / {sample_label(n_samples)}' for n_trees, n_samples in configurations]
    missing_keys = [name for name in config_names if name not in score_store_if]
    if missing_keys:
        raise KeyError(f'Missing configurations in score_store_if: {missing_keys}')
    score_arrays = [np.asarray(score_store_if[name], dtype=np.float64) for name in config_names]
    n_configs = len(score_arrays)
    n_observations = len(score_arrays[0])
    n_top = max(1, int(np.ceil(n_observations * 0.01)))
    spearman_matrix = np.eye(n_configs)
    top_sets = [set(np.argpartition(scores, -n_top)[-n_top:]) for scores in score_arrays]
    jaccard_matrix = np.eye(n_configs)
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            rho = spearmanr(score_arrays[i], score_arrays[j]).statistic
            intersection = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            jaccard = intersection / union if union else np.nan
            spearman_matrix[i, j] = rho
            spearman_matrix[j, i] = rho
            jaccard_matrix[i, j] = jaccard
            jaccard_matrix[j, i] = jaccard

    def plot_heatmap(matrix, labels, title, filename):
        fig, ax = plt.subplots(figsize=(15, 13), constrained_layout=True)
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = matrix[i, j]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6, color='white' if value < 0.65 else 'black')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Similarity')
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    plot_heatmap(spearman_matrix, config_labels, 'Isolation Forest - Spearman Ranking Correlation', 'if_spearman_heatmap.png')
    plot_heatmap(jaccard_matrix, config_labels, 'Isolation Forest - Top-1% Jaccard Similarity', 'if_jaccard_heatmap.png')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    tree_labels = [f'{tree_counts[i]} vs {tree_counts[i + 1]}' for i in range(len(tree_counts) - 1)]
    for n_samples in sample_sizes:
        spearman_values = []
        jaccard_values = []
        for i in range(len(tree_counts) - 1):
            idx1 = configurations.index((tree_counts[i], n_samples))
            idx2 = configurations.index((tree_counts[i + 1], n_samples))
            spearman_values.append(spearman_matrix[idx1, idx2])
            jaccard_values.append(jaccard_matrix[idx1, idx2])
        axes[0].plot(tree_labels, spearman_values, marker='o', linewidth=2, label=sample_label(n_samples))
        axes[1].plot(tree_labels, jaccard_values, marker='o', linewidth=2, label=sample_label(n_samples))
    axes[0].set_title('Spearman (Higher = More Stable)')
    axes[1].set_title('Top-1% Jaccard (Higher = More Stable)')
    for ax in axes:
        ax.set_xlabel('Number of Trees Comparison')
        ax.set_ylabel('Similarity')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[1].legend(title='Max Samples')
    fig.suptitle('Isolation Forest - Sensitivity to Number of Trees', fontsize=14, fontweight='bold')
    fig.savefig(output_dir / 'if_tree_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), constrained_layout=True)
    sample_labels = [f'{sample_label(sample_sizes[i])} vs {sample_label(sample_sizes[i + 1])}' for i in range(len(sample_sizes) - 1)]
    for n_trees in tree_counts:
        spearman_values = []
        jaccard_values = []
        for i in range(len(sample_sizes) - 1):
            idx1 = configurations.index((n_trees, sample_sizes[i]))
            idx2 = configurations.index((n_trees, sample_sizes[i + 1]))
            spearman_values.append(spearman_matrix[idx1, idx2])
            jaccard_values.append(jaccard_matrix[idx1, idx2])
        axes[0].plot(sample_labels, spearman_values, marker='o', linewidth=2, label=f'{n_trees} trees')
        axes[1].plot(sample_labels, jaccard_values, marker='o', linewidth=2, label=f'{n_trees} trees')
    axes[0].set_title('Spearman (Higher = More Stable)')
    axes[1].set_title('Top-1% Jaccard (Higher = More Stable)')
    for ax in axes:
        ax.set_xlabel('Sample Size Comparison')
        ax.set_ylabel('Similarity')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(alpha=0.3)
    axes[1].legend(title='Number of Trees')
    fig.suptitle('Isolation Forest - Sensitivity to Sample Size', fontsize=14, fontweight='bold')
    fig.savefig(output_dir / 'if_sample_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('\nAll Isolation Forest visualizations saved successfully.')


def plot_if_stability(experiment):
    spearman_matrix = experiment.spearman_matrix
    jaccard_matrix = experiment.jaccard_matrix
    output_dir = experiment.results_dir

    def plot_seed_heatmap(matrix, title, filename):
        values = matrix.to_numpy()
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        im = ax.imshow(values, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel('Random Seed')
        ax.set_ylabel('Random Seed')
        ax.set_title(title, fontsize=13, fontweight='bold')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9, color='white' if value < 0.65 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Similarity')
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    plot_seed_heatmap(spearman_matrix, 'Isolation Forest - Spearman Stability Across Seeds', 'if_seed_spearman_heatmap.png')
    plot_seed_heatmap(jaccard_matrix, 'Isolation Forest - Top-1% Jaccard Across Seeds', 'if_seed_jaccard_heatmap.png')
    print('\nIsolation Forest stability heatmaps saved successfully.')


