"""VAE seed experiments and original-unit anomaly interpretation for notebook 07."""
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from src.models.neural_networks import SEEDS, train_vae, vae_scores, save_table


class VAESeedTrainer:
    def __init__(self, data):
        self.data = data

    def train(self, config, seed, max_epochs=100, patience=10):
        return train_vae(self.data, config, seed=seed, max_epochs=max_epochs,
                         patience=patience, batch_size=256, independent_shuffle=True)

    def describe(self):
        print('VAE seed training: fixed architecture and split; independent seeded minibatches, Adam, patience 10.')


class VAESeedStability:
    def __init__(self, seeds=SEEDS, force_retrain=False):
        self.seeds, self.force_retrain = list(seeds), force_retrain

    def _validate(self):
        expected = {frozenset(pair) for pair in combinations(self.seeds, 2)}
        pairs = [frozenset((int(row.seed_a), int(row.seed_b))) for row in self.pairwise.itertuples()]
        if len(pairs) != len(expected) or set(pairs) != expected:
            raise ValueError('Incomplete or duplicated VAE seed comparisons.')
        for col in ('spearman','jaccard_05','jaccard_01'):
            if col not in self.pairwise or not np.isfinite(self.pairwise[col]).all():
                raise ValueError(f'Missing/invalid VAE stability metric: {col}')

    def run(self, search, trainer=None, max_epochs=100, patience=10):
        self.data, self.config = search.data, dict(search.config)
        self.results_dir = self.data.results_dir
        path = self.results_dir / 'vae_seed_pairwise_stability.csv'
        if path.exists() and not self.force_retrain:
            print('Existing VAE pairwise stability results found. Loading...')
            metrics_path = self.results_dir / 'vae_seed_stability.csv'
            if metrics_path.exists():
                saved = pd.read_csv(metrics_path)
                if saved.seed.duplicated().any() or set(saved.seed) != set(self.seeds):
                    raise ValueError('VAE seed metadata mismatch.')
                checks = {**self.config, 'split_seed':self.data.seed,
                          'train_rows':len(self.data.train), 'validation_rows':len(self.data.validation)}
                for key, value in checks.items():
                    if key in saved and not saved[key].eq(value).all():
                        raise ValueError(f'VAE seed experiment {key} mismatch.')
            self.pairwise = pd.read_csv(path)
        else:
            if self.force_retrain:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                self.results_dir = self.results_dir / 'vae_seed_stability' / stamp
                self.results_dir.mkdir(parents=True)
                path = self.results_dir / path.name
            trainer = trainer or VAESeedTrainer(self.data)
            settings = dict(config=self.config, data_fingerprint=self.data.fingerprint,
                seeds=self.seeds, max_epochs=max_epochs, patience=patience,
                batch_size=256, scoring='posterior_mean_reconstruction_MSE')
            fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            (self.results_dir / 'vae_stability_manifest.json').write_text(json.dumps(settings, indent=2))
            scores, thresholds, rows = {}, {}, []
            for seed in tqdm(self.seeds, desc='VAE seed stability', unit='seed'):
                score_path = self.results_dir / f'vae_seed_{seed}_scores.npz'
                if score_path.exists():
                    with np.load(score_path, allow_pickle=False) as cached:
                        if cached['fingerprint'].item() != fingerprint:
                            raise ValueError('Cached VAE scores belong to another experiment.')
                        train_scores, values = cached['train_scores'], cached['validation_scores']
                    if len(train_scores) != len(self.data.train) or len(values) != len(self.data.validation):
                        raise ValueError('Cached VAE score length mismatch.')
                else:
                    model, metrics, history = trainer.train(self.config, seed, max_epochs, patience)
                    values = vae_scores(model, self.data.validation_array)
                    train_scores = vae_scores(model, self.data.train_array)
                    with score_path.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream, fingerprint=np.array(fingerprint),
                                            train_scores=train_scores, validation_scores=values)
                    score_path.with_suffix('.tmp').replace(score_path)
                    save_table(pd.DataFrame(history), self.results_dir / f'vae_history_seed_{seed}.csv', index=False)
                    del model
                thresholds[seed] = {a:float(np.quantile(train_scores,1-a)) for a in (.05,.01)}
                scores[seed] = values
                rows.append(dict(seed=seed, threshold_05=thresholds[seed][.05], threshold_01=thresholds[seed][.01]))
            comparisons = []
            for a,b in combinations(self.seeds,2):
                row = dict(seed_a=a, seed_b=b, spearman=float(spearmanr(scores[a],scores[b]).statistic))
                for alpha,suffix in [(.05,'05'),(.01,'01')]:
                    fa,fb = scores[a]>thresholds[a][alpha],scores[b]>thresholds[b][alpha]
                    union = np.logical_or(fa,fb).sum()
                    row[f'jaccard_{suffix}'] = float(np.logical_and(fa,fb).sum()/union) if union else 1.
                comparisons.append(row)
            self.pairwise = pd.DataFrame(comparisons)
            save_table(self.pairwise,path,index=False)
            save_table(pd.DataFrame(rows),self.results_dir/'vae_seed_training_metrics.csv',index=False)
            frame = pd.DataFrame({f'seed_{seed}':values for seed,values in scores.items()},index=self.data.validation.index)
            save_table(frame,self.results_dir/'vae_seed_validation_scores.csv',index_label='source_index')
        self._validate()
        self.summary = pd.DataFrame([dict(Metric=title,Mean=self.pairwise[col].mean(),
            SD=self.pairwise[col].std(),Minimum=self.pairwise[col].min(),Maximum=self.pairwise[col].max())
            for title,col in [('Global Spearman','spearman'),('Jaccard (α = 0.05)','jaccard_05'),('Jaccard (α = 0.01)','jaccard_01')]])
        return self

    def display(self):
        from IPython.display import display
        print('Pairwise comparisons:',len(self.pairwise))
        display(self.pairwise)
        display(self.pairwise[['spearman','jaccard_05','jaccard_01']].agg(['mean','std','min','max']).round(4))


class VAEAnomalyAnalysis:
    def __init__(self, analysis, alpha=.05):
        if not 0 < alpha < 1:
            raise ValueError('alpha must lie between zero and one.')
        self.analysis, self.alpha = analysis, alpha
        self.threshold = float(np.quantile(analysis.train_scores,1-alpha))
        self.flags = analysis.validation_scores > self.threshold
        self.anomalous = analysis.data.validation.iloc[np.flatnonzero(self.flags)].copy()
        self.normal = analysis.data.validation.iloc[np.flatnonzero(~self.flags)].copy()

    def present(self):
        from IPython.display import display
        import matplotlib.pyplot as plt
        ALPHA = self.alpha
        percentile = int(round((1-ALPHA)*100))
        train_scores = self.analysis.train_scores
        val_scores = self.analysis.validation_scores
        X_val = self.analysis.data.validation

        threshold = np.quantile(train_scores, 1 - ALPHA)
        scores = np.asarray(val_scores).reshape(-1)
        assert len(X_val) == len(scores), 'X_val and val_scores must contain the same observations in the same order.'
        assert np.isfinite(scores).all(), 'Validation scores must be finite.'
        flags = scores > threshold
        anomalous = X_val.iloc[np.flatnonzero(flags)].copy()
        normal = X_val.iloc[np.flatnonzero(~flags)].copy()
        anomalous_scores = scores[flags]
        normal_scores = scores[~flags]
        print('=' * 75)
        print(f'VAE | {percentile}TH PERCENTILE | VALIDATION')
        print('=' * 75)
        print(f'Threshold: {threshold:.6f}')
        print(f'Total observations: {len(scores):,}')
        print(f'Anomalous: {flags.sum():,}')
        print(f'Non-anomalous: {(~flags).sum():,}')
        print(f'Anomaly rate: {flags.mean():.2%}')
        feature_statistics = pd.concat({'Anomalous': anomalous.describe().T, 'Non-Anomalous': normal.describe().T}, names=['Group', 'Feature'])
        print('\nVAE - ORIGINAL FEATURE DESCRIPTIVE STATISTICS')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(feature_statistics.round(4))
        score_statistics = pd.DataFrame({'Anomalous': pd.Series(anomalous_scores, dtype=float).describe(), 'Non-Anomalous': pd.Series(normal_scores, dtype=float).describe()})
        print('\nVAE - ANOMALY SCORE DESCRIPTIVE STATISTICS')
        display(score_statistics.round(6))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.boxplot([normal_scores, anomalous_scores], showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 7})
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Non-Anomalous', 'Anomalous'])
        ax.axhline(threshold, color='tab:blue', linestyle='--', linewidth=2, label=f'Training {percentile}th percentile = {threshold:.6f}')
        ax.set_title(f'VAE Anomaly Score Boxplot ({percentile}th Percentile)')
        ax.set_ylabel('VAE Reconstruction MSE')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.histogram_bin_edges(scores, bins=80)
        ax.hist(normal_scores, bins=bins, alpha=0.65, label='Non-Anomalous', color='tab:blue')
        ax.hist(anomalous_scores, bins=bins, alpha=0.65, label='Anomalous', color='tab:orange')
        ax.axvline(threshold, color='tab:blue', linestyle='--', linewidth=2, label=f'Training {percentile}th percentile = {threshold:.6f}')
        ax.set_title(f'VAE Anomaly Score Histogram ({percentile}th Percentile)')
        ax.set_xlabel('VAE Reconstruction MSE')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)
