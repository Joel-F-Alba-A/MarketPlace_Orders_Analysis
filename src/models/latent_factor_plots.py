"""Notebook 04 visualizations; model fitting lives in the model classes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from .latent_factors import NOTEBOOK_RESULTS

def plot_fa_search(search):
    fa_results = search.results
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    k_values = fa_results['k'].to_numpy()

    def annotate_values(ax, x, y, color, offset=8):
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.annotate(f'{yi:.4f}', (xi, yi), textcoords='offset points', xytext=(0, offset), ha='center', fontsize=8, color=color)
    ax = axes[0]
    ax.plot(k_values, fa_results['Train MSE'], marker='o', label='Training')
    ax.plot(k_values, fa_results['Validation MSE'], marker='s', label='Validation')
    annotate_values(ax, k_values, fa_results['Train MSE'], 'tab:blue', offset=9)
    annotate_values(ax, k_values, fa_results['Validation MSE'], 'tab:orange', offset=-15)
    ax.set_title('Reconstruction MSE')
    ax.set_ylabel('MSE')
    ax.legend()
    ax = axes[1]
    ax.plot(k_values, fa_results['Train Relative Frobenius'], marker='o', label='Training')
    ax.plot(k_values, fa_results['Validation Relative Frobenius'], marker='s', label='Validation')
    annotate_values(ax, k_values, fa_results['Train Relative Frobenius'], 'tab:blue', offset=9)
    annotate_values(ax, k_values, fa_results['Validation Relative Frobenius'], 'tab:orange', offset=-15)
    ax.set_title('Relative Reconstruction Error')
    ax.set_ylabel('Relative Frobenius Error')
    ax.legend()
    ax = axes[2]
    ax.plot(k_values, fa_results['Train Reconstructed Energy'], marker='o', label='Training')
    ax.plot(k_values, fa_results['Validation Reconstructed Energy'], marker='s', label='Validation')
    annotate_values(ax, k_values, fa_results['Train Reconstructed Energy'], 'tab:blue', offset=9)
    annotate_values(ax, k_values, fa_results['Validation Reconstructed Energy'], 'tab:orange', offset=-15)
    ax.set_title('Reconstructed Energy')
    ax.set_ylabel('Reconstructed Energy Ratio')
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax = axes[3]
    mean = fa_results['Stability Mean'].to_numpy()
    std = fa_results['Stability Std'].to_numpy()
    ax.errorbar(k_values, mean, yerr=std, fmt='o-', capsize=3, label='Mean ± 1 SD')
    ax.fill_between(k_values, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)
    annotate_values(ax, k_values, mean, 'tab:blue')
    ax.set_title('Factor Stability')
    ax.set_ylabel('Mean Absolute Cosine Similarity')
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax = axes[4]
    ax.plot(k_values, fa_results['Mean Concentration'], marker='o', label='Mean Concentration')
    annotate_values(ax, k_values, fa_results['Mean Concentration'], 'tab:blue')
    ax.set_title('Factor Concentration')
    ax.set_ylabel('Mean HHI Concentration')
    ax.legend()
    ax = axes[5]
    ax.plot(k_values, fa_results['Silhouette'], marker='o', label='Silhouette')
    annotate_values(ax, k_values, fa_results['Silhouette'], 'tab:blue')
    ax.set_title('Clustering Utility')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    for ax in axes:
        ax.set_xlabel('Number of Components (k)')
        ax.set_xticks(k_values)
        ax.grid(alpha=0.25, linestyle='--')
        ax.margins(x=0.04)
        ax.set_ylim(bottom=0)
        current_top = ax.get_ylim()[1]
        ax.set_ylim(top=current_top * 1.08)
    fig.suptitle('Factor Analysis — Evaluation Metrics Across Components', fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_ica_search(search):
    ica_results = search.results
    fig, axes = plt.subplots(2, 4, figsize=(25, 11))
    axes = axes.flatten()
    k_values = ica_results['k'].to_numpy()

    def annotate_values(ax, x, y, color, offset=8):
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.annotate(f'{yi:.4f}', (xi, yi), textcoords='offset points', xytext=(0, offset), ha='center', fontsize=8, color=color)
    ax = axes[0]
    ax.plot(k_values, ica_results['Train MSE'], marker='o', label='Training')
    ax.plot(k_values, ica_results['Validation MSE'], marker='s', label='Validation')
    annotate_values(ax, k_values, ica_results['Train MSE'], 'tab:blue', offset=9)
    annotate_values(ax, k_values, ica_results['Validation MSE'], 'tab:orange', offset=-15)
    ax.set_title('Reconstruction MSE')
    ax.set_ylabel('MSE')
    ax.legend()
    ax = axes[1]
    ax.plot(k_values, ica_results['Train Relative Frobenius'], marker='o', label='Training')
    ax.plot(k_values, ica_results['Validation Relative Frobenius'], marker='s', label='Validation')
    annotate_values(ax, k_values, ica_results['Train Relative Frobenius'], 'tab:blue', offset=9)
    annotate_values(ax, k_values, ica_results['Validation Relative Frobenius'], 'tab:orange', offset=-15)
    ax.set_title('Relative Reconstruction Error')
    ax.set_ylabel('Relative Frobenius Error')
    ax.legend()
    ax = axes[2]
    ax.plot(k_values, ica_results['Train Reconstructed Energy'], marker='o', label='Training')
    ax.plot(k_values, ica_results['Validation Reconstructed Energy'], marker='s', label='Validation')
    annotate_values(ax, k_values, ica_results['Train Reconstructed Energy'], 'tab:blue', offset=9)
    annotate_values(ax, k_values, ica_results['Validation Reconstructed Energy'], 'tab:orange', offset=-15)
    ax.set_title('Reconstructed Energy')
    ax.set_ylabel('Reconstructed Energy Ratio')
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax = axes[3]
    mean = ica_results['Stability Mean'].to_numpy()
    std = ica_results['Stability Std'].to_numpy()
    ax.errorbar(k_values, mean, yerr=std, fmt='o-', capsize=3, label='Mean ± 1 SD')
    ax.fill_between(k_values, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)
    annotate_values(ax, k_values, mean, 'tab:blue')
    ax.set_title('Component Stability')
    ax.set_ylabel('Mean Absolute Cosine Similarity')
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax = axes[4]
    ax.plot(k_values, ica_results['Mean Concentration'], marker='o', label='Mean Concentration')
    annotate_values(ax, k_values, ica_results['Mean Concentration'], 'tab:blue')
    ax.set_title('Component Concentration')
    ax.set_ylabel('Mean HHI Concentration')
    ax.legend()
    ax = axes[5]
    ax.plot(k_values, ica_results['Silhouette'], marker='o', label='Silhouette')
    annotate_values(ax, k_values, ica_results['Silhouette'], 'tab:blue')
    ax.set_title('Clustering Utility')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    ax = axes[6]
    ax.plot(k_values, ica_results['Mean Absolute Kurtosis'], marker='o', label='Mean Absolute Kurtosis')
    annotate_values(ax, k_values, ica_results['Mean Absolute Kurtosis'], 'tab:blue')
    ax.set_title('Component Non-Gaussianity')
    ax.set_ylabel('Mean Absolute Excess Kurtosis')
    ax.legend()
    ax = axes[7]
    ax.plot(k_values, ica_results['Iterations'], marker='o', label='Iterations')
    annotate_values(ax, k_values, ica_results['Iterations'], 'tab:blue')
    ax.set_title('ICA Convergence')
    ax.set_ylabel('Number of Iterations')
    ax.legend()
    for ax in axes:
        ax.set_xlabel('Number of Components (k)')
        ax.set_xticks(k_values)
        ax.grid(alpha=0.25, linestyle='--')
        ax.margins(x=0.04)
        ax.set_ylim(bottom=0)
        current_top = ax.get_ylim()[1]
        ax.set_ylim(top=current_top * 1.08)
    fig.suptitle('Independent Component Analysis (ICA) — Evaluation Metrics Across Components', fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_fa_loadings(analysis):
    fa_loadings = analysis.loadings
    K = analysis.k
    fig, ax = plt.subplots(figsize=(max(7, K * 1.8), max(5, len(fa_loadings) * 0.5)))
    limit = abs(fa_loadings.to_numpy()).max()
    sns.heatmap(fa_loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-limit, vmax=limit, linewidths=0.5, cbar_kws={'label': 'Factor loading'}, ax=ax)
    ax.set_title(f'Factor Analysis — Loadings (k={K})')
    ax.set_xlabel('Latent factors')
    ax.set_ylabel('Features')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    fig.savefig(analysis.search.cache_dir / f'fa_k{K}_loadings_heatmap.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_ica_loadings(analysis):
    ica_mixing = analysis.loadings
    fig, ax = plt.subplots(figsize=(13, max(7, len(ica_mixing) * 0.55)))
    limit = np.abs(ica_mixing.to_numpy()).max()
    sns.heatmap(ica_mixing, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-limit, vmax=limit, linewidths=0.5, cbar_kws={'label': 'Mixing coefficient'}, ax=ax)
    ax.set_title('Independent Component Analysis — Mixing Matrix (k={analysis.k})', fontsize=15, fontweight='bold')
    ax.set_xlabel('Independent Components')
    ax.set_ylabel('Original Features')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    plt.show()

def plot_factor_comparison(comparison):
    fa_loadings = comparison.fa.loadings
    ica_mixing = comparison.ica.loadings
    pca_loadings = comparison.pca_loadings
    feature_order = fa_loadings.index
    tables = [('FA — Factor Loadings', fa_loadings), ('ICA — Mixing Matrix', ica_mixing), ('PCA — Component Loadings', pca_loadings)]
    fig, axes = plt.subplots(1, 3, figsize=(27, max(8, len(feature_order) * 0.6)), constrained_layout=True)
    for ax, (title, table) in zip(axes, tables):
        missing = feature_order.difference(table.index)
        if not missing.empty:
            raise ValueError(f'{title}: missing features {missing.tolist()}')
        table = table.loc[feature_order]
        limit = np.abs(table.to_numpy()).max()
        sns.heatmap(table, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-limit, vmax=limit, linewidths=0.5, cbar_kws={'label': 'Coefficient'}, ax=ax)
        ax.set_title(f'{title} (k={table.shape[1]})', fontsize=14)
        ax.set_xlabel('Components')
        ax.set_ylabel('Features')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
    output_dir = NOTEBOOK_RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fa_ica_pca_heatmaps.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_recommendation_search(search):
    svd_results = search.results
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(svd_results['k'], svd_results['HitRate@10'], marker='o', label='HitRate@10')
    ax.plot(svd_results['k'], svd_results['NDCG@10'], marker='s', label='NDCG@10')
    ax.set_xlabel('Number of Latent Factors (k)')
    ax.set_ylabel('Validation Score')
    ax.set_title('Truncated SVD: Recommendation Performance')
    ax.legend()
    ax.grid(alpha=0.3)
    max_k = int(svd_results['k'].max())
    ax.set_xticks(np.arange(5, max_k + 1, 5))
    plt.show()

def plot_product_popularity(recommender):
    recommender.popularity_summary()
    product_popularity = recommender.popularity
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(product_popularity) + 1), product_popularity.values)
    plt.xlabel('Products ranked by popularity')
    plt.ylabel('Number of distinct customers')
    plt.title('Product Popularity Distribution')
    plt.tight_layout()
    plt.show()
