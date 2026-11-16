"""Notebook 07 plots; data preparation and model fitting live in their own classes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from src.models.neural_networks import save_table


def plot_ae_search(search):
    best_per_dimension = search.best_per_dimension
    output_dir = search.data.results_dir
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(best_per_dimension['latent_dim'], best_per_dimension['train_mse'], marker='o', label='Training MSE')
    ax.plot(best_per_dimension['latent_dim'], best_per_dimension['val_mse'], marker='o', label='Validation MSE')
    ax.set_xlabel('Number of Latent Dimensions')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Autoencoder — Reconstruction vs Latent Dimension')
    ax.set_xticks([2, 3, 4, 5])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ae_reconstruction_vs_latent_dim.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_ae_correlations(analysis):
    latent_correlations = analysis.correlations
    output_dir = analysis.data.results_dir
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(latent_correlations, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'label': 'Spearman correlation'}, ax=ax)
    ax.set_title('AE — Associations Between Original Features and Latent Variables', pad=15)
    ax.set_xlabel('Latent Variable')
    ax.set_ylabel('Original Feature')
    ax.tick_params(axis='x', labelrotation=0)
    ax.tick_params(axis='y', labelrotation=0)
    plt.tight_layout()
    fig.savefig(output_dir / 'ae_latent_feature_correlations.png', dpi=300, bbox_inches='tight')
    latent_correlations.to_csv(output_dir / 'ae_latent_feature_correlations.csv', index_label='Feature')
    plt.show()
    plt.close(fig)


def plot_vae_history(search):
    best_history = search.history
    history_df = pd.DataFrame(best_history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_df['epoch'], history_df['train_loss'], label='Training loss')
    ax.plot(history_df['epoch'], history_df['val_loss'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VAE loss')
    ax.set_title('VAE Training and Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_vae_score_distribution(analysis):
    val_scores = analysis.validation_scores
    thresholds = analysis.thresholds
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(val_scores, bins=80, alpha=0.7, label='Validation scores')
    ax.axvline(thresholds[0.05], linestyle='--', label='95th percentile threshold')
    ax.axvline(thresholds[0.01], linestyle=':', label='99th percentile threshold')
    ax.set_xlabel('VAE Reconstruction MSE')
    ax.set_ylabel('Number of observations')
    ax.set_title('VAE Anomaly Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ae_stability(experiment):
    return experiment.plot()


def plot_vae_stability(experiment):
    seeds, pairwise = experiment.seeds, experiment.pairwise
    for title, column in [('Global Spearman','spearman'), ('Jaccard (α = 0.05)','jaccard_05'), ('Jaccard (α = 0.01)','jaccard_01')]:
        matrix = pd.DataFrame(np.nan,index=seeds,columns=seeds,dtype=float)
        for seed in seeds:
            matrix.loc[seed,seed] = 1.
        for row in pairwise.itertuples(index=False):
            matrix.loc[row.seed_a,row.seed_b] = getattr(row,column)
            matrix.loc[row.seed_b,row.seed_a] = getattr(row,column)
        if matrix.isna().any().any():
            raise ValueError('Incomplete VAE stability matrix.')
        fig,ax = plt.subplots(figsize=(10,8))
        sns.heatmap(matrix,annot=True,fmt='.3f',cmap='viridis',vmin=0,vmax=1,
                    square=True,linewidths=.5,linecolor='white',cbar_kws={'label':'Pairwise Stability'},ax=ax)
        ax.set(title=f'VAE — {title} Across Random Seeds',xlabel='Random Seed',ylabel='Random Seed')
        ax.tick_params(axis='both',labelrotation=0)
        fig.tight_layout()
        fig.savefig(experiment.results_dir/f'vae_{column}_pairwise_heatmap.png',dpi=300,bbox_inches='tight')
        matrix.to_csv(experiment.results_dir/f'vae_{column}_pairwise_matrix.csv')
        plt.show()
        plt.close(fig)
    print('VAE SEED STABILITY SUMMARY')
    display(experiment.summary.round(4))
    save_table(experiment.summary,experiment.results_dir/'vae_pairwise_stability_summary.csv',index=False)
