"""Load or refit AE seed experiments and plot latent representation stability."""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.config import CLEAN_DATA_PATH, RESULTS_DIR


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_csv(frame, path, **kwargs):
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def _cka(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = np.linalg.norm(x.T @ x, 'fro') * np.linalg.norm(y.T @ y, 'fro')
    return float(np.linalg.norm(x.T @ y, 'fro') ** 2 / denominator) if denominator else np.nan


class AESeedStability:
    """Load an organized AE experiment, optionally refit, then compare representations."""

    def __init__(self, results_dir=None, dataset_path=CLEAN_DATA_PATH, experiment=None):
        self.base = Path(results_dir or RESULTS_DIR / '07_autoencoder_vae') / 'seed_stability'
        if experiment is None:
            experiment = (self.base / 'latest_run.txt').read_text().strip()
        self.directory = self.base / experiment
        self.settings = json.loads((self.directory / 'manifest.json').read_text())
        self.dataset_path = Path(dataset_path)
        self.config = self.settings['ae_config']
        self.seeds = self.settings['seeds']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('The manifest must contain at least two distinct seeds.')

    def _prepare(self):
        if _sha(self.dataset_path) != self.settings['dataset_sha256']:
            raise ValueError('The dataset differs from the dataset used in Colab.')
        frame = pd.read_csv(self.dataset_path)
        x = frame[self.settings['features']].copy()
        train, temporary = train_test_split(x, test_size=.4, random_state=self.settings['split_seed'])
        validation, _ = train_test_split(temporary, test_size=.5, random_state=self.settings['split_seed'])
        scaler = StandardScaler().fit(train)
        self.scaler = scaler
        self.train = np.asarray(scaler.transform(train), dtype=np.float32)
        self.validation = np.asarray(scaler.transform(validation), dtype=np.float32)
        self.validation_indices = validation.index.to_numpy()
        if not np.isfinite(self.train).all() or not np.isfinite(self.validation).all():
            raise ValueError('Non-finite numerical features.')
        saved_scaler = self.directory / 'scaler.csv'
        if saved_scaler.exists():
            saved = pd.read_csv(saved_scaler)
            if list(saved.feature) != self.settings['features']:
                raise ValueError('Scaler feature order mismatch.')
            np.testing.assert_allclose(saved.training_mean, scaler.mean_, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(saved.training_scale, scaler.scale_, rtol=1e-10, atol=1e-10)
        count = min(self.settings['evaluation_size'], len(validation))
        self.evaluation_indices = np.sort(np.random.default_rng(self.settings['evaluation_seed']).choice(
            len(validation), size=count, replace=False))
        saved_indices = self.directory / 'ae_evaluation_indices.csv'
        if saved_indices.exists():
            saved = pd.read_csv(saved_indices)
            np.testing.assert_array_equal(saved.validation_position, self.evaluation_indices)
            np.testing.assert_array_equal(saved.source_index, self.validation_indices[self.evaluation_indices])

    def _new_run(self):
        original = self.directory
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.directory = self.base / f'ae_local_refit_{stamp}'
        self.directory.mkdir(parents=True)
        self.settings = {**self.settings, 'origin_experiment': original.name,
                         'torch_version': str(torch.__version__),
                         'device_names': [str(self.device)],
                         'training_implementation': 'AE training callable',
                         'deterministic': torch.are_deterministic_algorithms_enabled()}
        (self.directory / 'manifest.json').write_text(json.dumps(self.settings, indent=2))
        for filename in ('scaler.csv', 'ae_evaluation_indices.csv'):
            if (original / filename).exists():
                (self.directory / filename).write_bytes((original / filename).read_bytes())
        # The original Colab experiment remains the default. Select a refit explicitly.

    def _check_training_inputs(self, train_fn):
        if train_fn is None:
            raise ValueError('A missing/refitted seed requires an AE training function.')
        trainer = getattr(train_fn, '__self__', None)
        for name, expected in [('X_train_tensor', self.train), ('X_val_tensor', self.validation)]:
            if trainer is not None and hasattr(trainer, 'data'):
                attr = 'train_tensor' if name == 'X_train_tensor' else 'validation_tensor'
                actual = getattr(trainer.data, attr)
            else:
                actual = train_fn.__globals__.get(name)
            if actual is None:
                raise ValueError(f'Prepare {name} before retraining.')
            actual = actual.detach().cpu().numpy()
            if actual.shape != expected.shape:
                raise ValueError(f'{name} does not match the saved split.')
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def display(self):
        from IPython.display import display
        print('RECONSTRUCTION METRICS BY SEED')
        display(self.seed_metrics.round(7))
        print('PAIRWISE REPRESENTATION STABILITY')
        display(self.pairwise.round(7))
        print('RECONSTRUCTION SUMMARY')
        display(self.reconstruction_summary.round(7))
        print('PAIRWISE STABILITY SUMMARY')
        display(self.summary.round(7))

    def _checkpoint(self, seed, train_fn):
        path = self.directory / f'ae_seed_{seed}.pt'
        if path.exists():
            value = torch.load(path, map_location='cpu', weights_only=True)
            if value.get('seed') != seed:
                raise ValueError(f'Checkpoint seed mismatch: {path.name}')
            if not value.get('complete', False):
                raise ValueError(f'{path.name} is incomplete. Resume it in Colab or use force_retrain=True.')
            metrics = value['metrics']
            if any(metrics.get(k) != v for k, v in self.config.items()):
                raise ValueError(f'Checkpoint hyperparameter mismatch: {path.name}')
            expected = hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest()
            if value.get('experiment_id') != expected:
                raise ValueError(f'Checkpoint experiment fingerprint mismatch: {path.name}')
            return path, value
        self._check_training_inputs(train_fn)
        model, metrics, history = train_fn(
            **self.config, seed=seed, max_epochs=self.settings['max_epochs'],
            patience=self.settings['patience'], min_delta=self.settings['min_delta'], show_progress=True)
        best = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
        metrics = {key: val.item() if isinstance(val, np.generic) else val for key, val in metrics.items()}
        fingerprint = hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest()
        value = dict(seed=seed, complete=True, experiment_id=fingerprint,
                     best_state=best, metrics={**metrics, **self.config, 'seed': seed})
        temporary = path.with_suffix('.tmp')
        torch.save(value, temporary)
        temporary.replace(path)
        _save_csv(history, self.directory / f'ae_history_seed_{seed}.csv', index=False)
        del model
        return path, value

    @torch.no_grad()
    def _representation(self, model):
        latent, reconstruction = [], []
        sample = self.validation[self.evaluation_indices]
        for first in range(0, len(sample), 2048):
            batch = torch.as_tensor(sample[first:first + 2048], device=self.device)
            z = model.encoder(batch)
            latent.append(z.cpu().numpy())
            reconstruction.append(model.decoder(z).cpu().numpy())
        return np.concatenate(latent), np.concatenate(reconstruction)

    def run(self, model_class, train_fn=None, force_retrain=False, recompute=False):
        """Load all completed results; refit only when explicitly requested/missing."""
        self._prepare()
        if force_retrain:
            self._check_training_inputs(train_fn)
            self._new_run()
        metrics_path = self.directory / 'ae_seed_stability.csv'
        pairs_path = self.directory / 'ae_seed_pairwise_stability.csv'
        need_outputs = force_retrain or recompute or not pairs_path.exists()
        rows, latent, reconstruction = [], {}, {}
        previous_threads = torch.get_num_threads()
        try:
            if self.device.type == 'cpu':
                torch.set_num_threads(min(4, previous_threads))
            for seed in tqdm(self.seeds, desc='AE seed results', unit='seed'):
                _, checkpoint = self._checkpoint(seed, train_fn)
                rows.append(checkpoint['metrics'])
                if need_outputs:
                    model = model_class(input_dim=self.train.shape[1], **{
                        key: self.config[key] for key in ('latent_dim', 'hidden_dim')}).to(self.device)
                    model.load_state_dict(checkpoint['best_state'])
                    model.eval()
                    latent[seed], reconstruction[seed] = self._representation(model)
                    target = self.directory / f'ae_outputs_seed_{seed}.npz'
                    with target.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream, experiment_id=np.array(checkpoint['experiment_id']),
                                            latent=latent[seed], reconstruction=reconstruction[seed])
                    target.with_suffix('.tmp').replace(target)
                    del model
                    gc.collect()
        finally:
            torch.set_num_threads(previous_threads)
        # The stored validation MSE is evaluated on all validation rows, not only
        # the fixed sample used for CKA and differences between reconstructions.
        self.seed_metrics = pd.DataFrame(rows).set_index('seed').loc[self.seeds].reset_index()
        if need_outputs:
            comparisons = [dict(seed_a=a, seed_b=b, linear_cka=_cka(latent[a], latent[b]),
                reconstruction_rmse=float(np.sqrt(np.mean((reconstruction[a] - reconstruction[b]) ** 2))))
                for a, b in combinations(self.seeds, 2)]
            _save_csv(pd.DataFrame(comparisons), pairs_path, index=False)
        self.pairwise = pd.read_csv(pairs_path)
        expected = set(combinations(self.seeds, 2))
        actual = {(int(row.seed_a), int(row.seed_b)) for row in self.pairwise.itertuples()}
        if actual != expected or len(self.pairwise) != len(expected):
            raise ValueError('Incomplete or duplicated representation comparisons.')
        values = self.pairwise[['linear_cka', 'reconstruction_rmse']].to_numpy()
        if (not np.isfinite(values).all() or (values[:, 0] < -1e-8).any()
                or (values[:, 0] > 1 + 1e-8).any() or (values[:, 1] < 0).any()):
            raise ValueError('Invalid representation stability metrics.')
        if not np.isfinite(self.seed_metrics[['train_mse', 'val_mse']]).all().all():
            raise ValueError('Invalid seed reconstruction metrics.')
        self.summary = self.pairwise[['linear_cka', 'reconstruction_rmse']].agg(['mean', 'std', 'min', 'max'])
        self.reconstruction_summary = self.seed_metrics[['train_mse', 'val_mse', 'best_epoch']].agg(['mean', 'std', 'min', 'max'])
        if need_outputs or not metrics_path.exists():
            _save_csv(self.seed_metrics, metrics_path, index=False)
            _save_csv(self.summary, self.directory / 'ae_pairwise_stability_summary.csv', index_label='statistic')
            _save_csv(self.reconstruction_summary, self.directory / 'ae_reconstruction_summary.csv', index_label='statistic')
        print(f'Validation MSE: all {len(self.validation):,} validation rows at the selected epoch.')
        print(f'CKA / pairwise RMSE: {len(self.evaluation_indices):,} identical validation rows per seed.')
        print(f'{len(self.seeds)} seeds | {len(self.pairwise)} unique pairs | Results: {self.directory}')
        return self

    def load_model(self, model_class, seed=42):
        """Load the validation-selected weights, not the last-epoch resume weights."""
        if not hasattr(self, 'train'):
            self._prepare()
        if seed not in self.seeds:
            raise ValueError('Seed is not part of this experiment.')
        _, checkpoint = self._checkpoint(seed, train_fn=None)
        model = model_class(input_dim=self.train.shape[1], **{
            key: self.config[key] for key in ('latent_dim', 'hidden_dim')}).to(self.device)
        model.load_state_dict(checkpoint['best_state'])
        model.eval()
        return model, checkpoint['metrics']

    def plot(self):
        """Validation MSE bars and the two latent-representation heatmaps."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        plots_dir = self.directory / 'plots'
        plots_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar([str(seed) for seed in self.seeds], self.seed_metrics.val_mse, color='#22648A')
        ax.bar_label(bars, labels=[f'{value:.2e}' for value in self.seed_metrics.val_mse],
                     padding=4, fontsize=9)
        ax.set(title='AE — validation reconstruction MSE across seeds', xlabel='Random seed',
               ylabel='Best validation MSE (lower is better)', ylim=(0, self.seed_metrics.val_mse.max() * 1.22))
        fig.tight_layout()
        fig.savefig(plots_dir / 'ae_validation_mse_by_seed.png', dpi=250, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        for metric, title, diagonal in [
            ('linear_cka', 'Latent linear CKA (higher is more similar)', 1.0),
            ('reconstruction_rmse', 'Pairwise reconstruction RMSE (lower is more similar)', 0.0),
        ]:
            matrix = pd.DataFrame(np.nan, index=self.seeds, columns=self.seeds)
            for seed in self.seeds:
                matrix.loc[seed, seed] = diagonal
            for row in self.pairwise.itertuples(index=False):
                matrix.loc[row.seed_a, row.seed_b] = getattr(row, metric)
                matrix.loc[row.seed_b, row.seed_a] = getattr(row, metric)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='.3f' if metric == 'linear_cka' else '.4f',
                        square=True, linewidths=.4, cmap='viridis' if metric == 'linear_cka' else 'magma_r',
                        vmin=0, vmax=1 if metric == 'linear_cka' else None, ax=ax)
            ax.set(title=f'AE — {title}', xlabel='Random seed', ylabel='Random seed')
            fig.tight_layout()
            fig.savefig(plots_dir / f'ae_{metric}_heatmap.png', dpi=250, bbox_inches='tight')
            matrix.to_csv(plots_dir / f'ae_{metric}_matrix.csv', index_label='seed')
            plt.show()
            plt.close(fig)
