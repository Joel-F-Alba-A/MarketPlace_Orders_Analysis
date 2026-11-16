"""Notebook 07 data, architectures, training and cached model searches.

Training objectives, initialization and minibatch order follow the original
notebook. Fitted scalers and latent frames never mutate the input DataFrame.
"""
import copy
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.config import CLEAN_DATA_PATH, RESULTS_DIR

FEATURES = [
    'Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order',
    'Sales per customer', 'Order Item Discount', 'Order Item Discount Rate',
    'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Product Price',
]
AE_CONFIG = dict(latent_dim=5, hidden_dim=64, learning_rate=.001, batch_size=128)
VAE_CONFIG = dict(latent_dim=5, hidden_dim=64, beta=.001, learning_rate=.001)
AE_GRID = dict(latent_dim=[2,3,4,5], hidden_dim=[16,32,64], learning_rate=[.0005,.001], batch_size=[128,256])
VAE_GRID = dict(latent_dim=[2,3,5], hidden_dim=[16,32,64], beta=[.001,.01,.1,1.], learning_rate=[.0005,.001])
SEEDS = [0,21,42,123,7,13,57,99,202,314]


def configure(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def plain(values):
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in values.items()}


def save_table(frame, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def save_checkpoint(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class NeuralData:
    def __init__(self, path=CLEAN_DATA_PATH, features=FEATURES, frame=None,
                 seed=42, device=None, results_dir=None):
        self.path = Path(path)
        self.frame = pd.read_csv(path) if frame is None else frame.copy(deep=True)
        self.features, self.seed = list(features), seed
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.results_dir = Path(results_dir or RESULTS_DIR / '07_autoencoder_vae')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_sha256 = (hashlib.sha256(self.path.read_bytes()).hexdigest() if frame is None
            else hashlib.sha256(pd.util.hash_pandas_object(self.frame, index=True).values.tobytes()).hexdigest())

    def select_features(self):
        self.X = self.frame[self.features].copy()
        if not np.isfinite(self.X.to_numpy(dtype=float)).all():
            raise ValueError('Numerical features must contain finite, non-missing values.')
        return self.X

    def prepare(self):
        if not hasattr(self, 'X'):
            self.select_features()
        self.train, temporary = train_test_split(self.X, test_size=.4, random_state=self.seed)
        self.validation, self.test = train_test_split(temporary, test_size=.5, random_state=self.seed)
        self.scaler = StandardScaler().fit(self.train)
        for split in ('train','validation','test'):
            scaled = self.scaler.transform(getattr(self, split))
            setattr(self, f'{split}_scaled', scaled)
            setattr(self, f'{split}_array', np.asarray(scaled, dtype=np.float32))
            setattr(self, f'{split}_tensor', torch.tensor(getattr(self, f'{split}_array')))
        self.input_dim = len(self.features)
        self.fingerprint = hashlib.sha256(json.dumps(dict(dataset=self.dataset_sha256,
            features=self.features, split_seed=self.seed, split=[.6,.2,.2]), sort_keys=True).encode()).hexdigest()
        return self

    def describe(self):
        print('Device:', self.device)
        print('Input features:', self.input_dim)
        print('Training observations:', len(self.train))
        print('Validation observations:', len(self.validation))




class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim=3, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return (reconstruction, mu, logvar)

def vae_loss(x, reconstruction, mu, logvar, beta=0.01):
    reconstruction_loss = (x - reconstruction).pow(2).mean()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return (total_loss, reconstruction_loss, kl_loss)

def train_ae(data, latent_dim=3, hidden_dim=32, learning_rate=0.001, batch_size=256, seed=42, max_epochs=100, patience=15, min_delta=1e-06, show_progress=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Autoencoder(input_dim=data.input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(data.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(data.train_tensor), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(data.validation_tensor), batch_size=1024, shuffle=False)
    best_val_mse = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []
    epoch_bar = tqdm(range(1, max_epochs + 1), desc=f'AE z={latent_dim}, h={hidden_dim}', leave=False, disable=not show_progress)
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = 0.0
        for batch, in train_loader:
            batch = batch.to(data.device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch.size(0)
        train_mse = train_loss_sum / len(data.train_tensor)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(data.device)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                val_loss_sum += loss.item() * batch.size(0)
        val_mse = val_loss_sum / len(data.validation_tensor)
        history.append({'epoch': epoch, 'train_mse': train_mse, 'val_mse': val_mse})
        if val_mse < best_val_mse - min_delta:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        epoch_bar.set_postfix({'Train': f'{train_mse:.5f}', 'Val': f'{val_mse:.5f}', 'Best': f'{best_val_mse:.5f}', 'Patience': f'{patience_counter}/{patience}'})
        if patience_counter >= patience:
            break
    epoch_bar.close()
    model.load_state_dict(best_state)
    model.eval()
    history_df = pd.DataFrame(history)
    best_metrics = {'train_mse': history_df.loc[history_df['epoch'] == best_epoch, 'train_mse'].iloc[0], 'val_mse': best_val_mse, 'best_epoch': best_epoch, 'epochs_trained': len(history_df)}
    return (model, best_metrics, history_df)

def train_vae(data, config, seed=42, max_epochs=100, patience=10, batch_size=256, independent_shuffle=False):
    train_dataset = TensorDataset(data.train_tensor)
    val_dataset = TensorDataset(data.validation_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    torch.manual_seed(seed)
    if independent_shuffle:
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed) if independent_shuffle else None
    model = VAE(input_dim=data.input_dim, latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim']).to(data.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        for batch, in train_loader:
            batch = batch.to(data.device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(batch, reconstruction, mu, logvar, beta=config['beta'])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch.size(0)
        train_loss = train_loss_sum / len(train_dataset)
        model.eval()
        val_loss_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(data.device)
                hidden = model.encoder(batch)
                mu = model.fc_mu(hidden)
                logvar = model.fc_logvar(hidden)
                reconstruction = model.decoder(mu)
                loss, recon_loss, kl_loss = vae_loss(batch, reconstruction, mu, logvar, beta=config['beta'])
                n = batch.size(0)
                val_loss_sum += loss.item() * n
                val_recon_sum += recon_loss.item() * n
                val_kl_sum += kl_loss.item() * n
        val_loss = val_loss_sum / len(val_dataset)
        val_recon = val_recon_sum / len(val_dataset)
        val_kl = val_kl_sum / len(val_dataset)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_reconstruction': val_recon, 'val_kl': val_kl})
        if val_loss < best_val_loss - 1e-06:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    model.load_state_dict(best_state)
    best_metrics = history[best_epoch - 1]
    return (model, best_metrics, history)


@dataclass
class TrainingResult:
    model: nn.Module
    metrics: dict
    history: pd.DataFrame

    def display(self):
        from IPython.display import display
        print('Training completed successfully.')
        print('\nMetrics:')
        display(pd.DataFrame([self.metrics]))
        print('\nTraining history:')
        display(self.history.tail())


class AETrainer:
    def __init__(self, data):
        self.data = data

    def train(self, **kwargs):
        return train_ae(self.data, **kwargs)

    def preview(self, latent_dim=3, hidden_dim=32):
        model = Autoencoder(self.data.input_dim, latent_dim, hidden_dim).to(self.data.device)
        with torch.no_grad():
            sample = self.data.train_tensor[:5].to(self.data.device)
            reconstruction, latent = model(sample), model.encoder(sample)
        print('Device:', self.data.device)
        print('Training shape:', self.data.train_tensor.shape)
        print('Validation shape:', self.data.validation_tensor.shape)
        print('Input dimensions:', self.data.input_dim)
        print('Latent shape:', latent.shape)
        print('Reconstruction shape:', reconstruction.shape)
        print('Trainable parameters:', sum(p.numel() for p in model.parameters()))

    def describe(self):
        print('AE training: Adam, reconstruction MSE, seeded minibatches and validation early stopping.')

    def demo(self, force_retrain=False):
        config = dict(latent_dim=3, hidden_dim=32, learning_rate=.001,
                      batch_size=256, seed=42, max_epochs=20, patience=5)
        path = self.data.results_dir / f'ae_training_demo_{self.data.fingerprint[:12]}.pt'
        if path.exists() and not force_retrain:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint['data_fingerprint'] != self.data.fingerprint:
                raise ValueError('AE training demonstration data mismatch.')
            model = Autoencoder(self.data.input_dim, 3, 32).to(self.data.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return TrainingResult(model, checkpoint['metrics'], pd.DataFrame(checkpoint['history']))
        model, metrics, history = self.train(**config)
        save_checkpoint(dict(data_fingerprint=self.data.fingerprint, config=config,
            model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            metrics=plain(metrics), history=history.to_dict('list')), path)
        return TrainingResult(model, metrics, history)


class AESearch:
    def __init__(self, grid=None, seed=42, max_epochs=100, patience=15, force_retrain=False):
        self.grid, self.seed = grid or AE_GRID, seed
        self.max_epochs, self.patience, self.force_retrain = max_epochs, patience, force_retrain

    def run(self, data):
        self.data = data
        path = data.results_dir / 'ae_hyperparameter_search.csv'
        keys = list(self.grid)
        configs = [dict(zip(keys, values)) for values in product(*(self.grid[k] for k in keys))]
        frame = pd.read_csv(path) if path.exists() and not self.force_retrain else pd.DataFrame()
        if not frame.empty:
            if frame[keys].duplicated().any():
                raise ValueError('Duplicate AE search configurations.')
            if any(not frame[key].isin(values).all() for key, values in self.grid.items()):
                raise ValueError('Cached AE configurations differ from the requested grid; use force_retrain=True.')
            if 'dataset_sha256' in frame and not frame.dataset_sha256.eq(data.dataset_sha256).all():
                raise ValueError('AE search belongs to another dataset.')
            for col, value in [('seed',self.seed), ('split_seed',data.seed),
                               ('max_epochs',self.max_epochs), ('patience',self.patience)]:
                if col in frame and not frame[col].eq(value).all():
                    raise ValueError(f'AE search {col} mismatch.')
            if not np.isfinite(frame[['train_mse','val_mse']]).all().all():
                raise ValueError('Invalid cached AE search metrics.')
        def key(row):
            return tuple(row[k] for k in keys)
        completed = {key(row) for row in frame.to_dict('records')}
        rows = frame.to_dict('records')
        print(f'Total configurations: {len(configs)} | Completed: {len(completed)}')
        for config in tqdm(configs, desc='AE configuration search', unit='model'):
            if key(config) in completed:
                continue
            started = time.perf_counter()
            model, metrics, history = train_ae(data, **config, seed=self.seed,
                max_epochs=self.max_epochs, patience=self.patience)
            name = f"ae_z{config['latent_dim']}_h{config['hidden_dim']}_lr{config['learning_rate']}_b{config['batch_size']}.pt"
            save_checkpoint(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                config=config, input_dim=data.input_dim, seed=self.seed,
                data_fingerprint=data.fingerprint, metrics=plain(metrics)), data.results_dir / 'candidate_models' / name)
            rows.append(dict(**config, **plain(metrics), seed=self.seed,
                training_seconds=time.perf_counter()-started, model_file=name,
                dataset_sha256=data.dataset_sha256, split_seed=data.seed,
                max_epochs=self.max_epochs, patience=self.patience))
            save_table(pd.DataFrame(rows), path, index=False)
        self.results = pd.DataFrame(rows).sort_values('val_mse').reset_index(drop=True)
        self.best_per_dimension = (self.results.groupby('latent_dim', as_index=False).first()
                                   .sort_values('latent_dim'))
        return self

    def display(self):
        from IPython.display import display
        display(self.results.head())

    def display_best_dimensions(self):
        from IPython.display import display
        columns = ['latent_dim','hidden_dim','learning_rate','batch_size','train_mse','val_mse','best_epoch','epochs_trained']
        display(self.best_per_dimension[columns].round(6))


@torch.inference_mode()
def ae_outputs(model, values, batch_size=2048):
    model.eval()
    device = next(model.parameters()).device
    latent, reconstruction = [], []
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start:start+batch_size], dtype=torch.float32, device=device)
        z = model.encoder(batch)
        latent.append(z.cpu().numpy())
        reconstruction.append(model.decoder(z).cpu().numpy())
    return np.concatenate(latent), np.concatenate(reconstruction)


class AEAnalysis:
    def __init__(self, data, config=None, seed=42):
        self.data, self.config, self.seed = data, dict(config or AE_CONFIG), seed

    def fit(self, stability=None, force_retrain=False):
        if stability is not None and not force_retrain:
            if stability.config != self.config:
                raise ValueError('Selected AE differs from the stability configuration.')
            self.model, self.metrics = stability.load_model(Autoencoder, self.seed)
        else:
            self.model, self.metrics, self.history = train_ae(self.data, **self.config, seed=self.seed)
        self.model.eval()
        self.latent = {}
        for split in ('train','validation'):
            values, _ = ae_outputs(self.model, getattr(self.data, f'{split}_array'))
            self.latent[split] = pd.DataFrame(values, index=getattr(self.data, split).index,
                columns=[f'AE {i+1}' for i in range(self.config['latent_dim'])])
            self.latent[split].index.name = 'source_index'
        joined = pd.concat([self.data.validation, self.latent['validation']], axis=1)
        self.correlations = joined.corr(method='spearman').loc[self.data.features, self.latent['validation'].columns]
        if force_retrain or not (self.data.results_dir / 'ae_best_model.pt').exists():
            self.save()
        return self

    def save(self):
        save_checkpoint(dict(
            model_state_dict={key:value.detach().cpu().clone() for key,value in self.model.state_dict().items()},
            config=self.config, input_dim=self.data.input_dim, seed=self.seed,
            features=self.data.features, scaler_mean=torch.as_tensor(self.data.scaler.mean_),
            scaler_scale=torch.as_tensor(self.data.scaler.scale_), data_fingerprint=self.data.fingerprint,
        ), self.data.results_dir / 'ae_best_model.pt')

    def display(self):
        from IPython.display import display
        print('Selected AE loaded from its best epoch.')
        print('\nHYPERPARAMETERS\n', self.config)
        display(pd.DataFrame([self.metrics]))
        print('Training latent shape:', self.latent['train'].shape)
        print('Validation latent shape:', self.latent['validation'].shape)
        display(self.latent['validation'].head())


class VAESearch:
    loss = staticmethod(vae_loss)

    def __init__(self, grid=None, seed=42, max_epochs=100, patience=10,
                 batch_size=256, force_retrain=False):
        self.grid, self.seed = grid or VAE_GRID, seed
        self.max_epochs, self.patience = max_epochs, patience
        self.batch_size, self.force_retrain = batch_size, force_retrain

    def run(self, data):
        self.data = data
        path = data.results_dir / 'vae_hyperparameter_search.csv'
        checkpoint_path = data.results_dir / 'vae_best_model.pt'
        history_path = data.results_dir / 'vae_training_history.csv'
        if path.exists() and checkpoint_path.exists() and history_path.exists() and not self.force_retrain:
            self.results = pd.read_csv(path).sort_values('val_loss').reset_index(drop=True)
            if self.results[list(self.grid)].duplicated().any() or not np.isfinite(self.results.val_loss).all():
                raise ValueError('Invalid cached VAE search.')
            expected_count = int(np.prod([len(values) for values in self.grid.values()]))
            if len(self.results) != expected_count or any(
                    not self.results[key].isin(values).all() for key,values in self.grid.items()):
                raise ValueError('Cached VAE search differs from the requested grid; use force_retrain=True.')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self.config = {k: int(self.results.iloc[0][k]) if k in ('latent_dim','hidden_dim')
                           else float(self.results.iloc[0][k]) for k in self.grid}
            if checkpoint['config'] != self.config or checkpoint['input_dim'] != data.input_dim:
                raise ValueError('VAE checkpoint/search configuration mismatch.')
            if checkpoint.get('data_fingerprint',data.fingerprint) != data.fingerprint:
                raise ValueError('VAE checkpoint data mismatch.')
            requested_settings = dict(seed=self.seed, max_epochs=self.max_epochs,
                                      patience=self.patience, batch_size=self.batch_size)
            if checkpoint.get('training_settings',requested_settings) != requested_settings:
                raise ValueError('VAE training settings differ; use force_retrain=True.')
            self.model = VAE(data.input_dim, self.config['latent_dim'], self.config['hidden_dim']).to(data.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.history = pd.read_csv(history_path)
            return self
        keys = list(self.grid)
        configs = [dict(zip(keys, values)) for values in product(*(self.grid[k] for k in keys))]
        rows, best = [], float('inf')
        for config in tqdm(configs, desc='VAE configuration search', unit='model'):
            model, metrics, history = train_vae(data, config, seed=self.seed,
                max_epochs=self.max_epochs, patience=self.patience, batch_size=self.batch_size)
            rows.append(dict(**config, best_epoch=metrics['epoch'],
                **{k:metrics[k] for k in ('train_loss','val_loss','val_reconstruction','val_kl')}))
            if metrics['val_loss'] < best:
                best = metrics['val_loss']
                self.model, self.config, self.history = model, dict(config), pd.DataFrame(history)
            save_table(pd.DataFrame(rows), path.with_name('vae_search_progress.csv'), index=False)
        self.results = pd.DataFrame(rows).sort_values('val_loss').reset_index(drop=True)
        self.save()
        return self

    def display(self):
        from IPython.display import display
        display(self.results.head(10))
        print('\nBest configuration:\n', self.config)
        print('\nBest validation loss:\n', self.results.iloc[0].val_loss)

    def save(self):
        save_table(self.results, self.data.results_dir / 'vae_hyperparameter_search.csv', index=False)
        save_table(self.history, self.data.results_dir / 'vae_training_history.csv', index=False)
        save_checkpoint(dict(model_state_dict={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            config=self.config, input_dim=self.data.input_dim, data_fingerprint=self.data.fingerprint,
            training_settings=dict(seed=self.seed,max_epochs=self.max_epochs,
                                   patience=self.patience,batch_size=self.batch_size)),
            self.data.results_dir / 'vae_best_model.pt')


@torch.inference_mode()
def vae_scores(model, values, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    tensor = torch.as_tensor(values, dtype=torch.float32)
    scores = []
    for batch in tensor.split(batch_size):
        batch = batch.to(device)
        mu = model.fc_mu(model.encoder(batch))
        reconstruction = model.decoder(mu)
        scores.append((batch-reconstruction).square().mean(dim=1).cpu().numpy())
    return np.concatenate(scores)


class VAEAnalysis:
    def __init__(self, search):
        self.search, self.data, self.model = search, search.data, search.model

    def score(self):
        self.train_scores = vae_scores(self.model, self.data.train_array)
        self.validation_scores = vae_scores(self.model, self.data.validation_array)
        print('Train scores:', self.train_scores.shape)
        print('Validation scores:', self.validation_scores.shape)
        return self

    def flag(self, alphas=(.05,.01)):
        from IPython.display import display
        self.thresholds = {alpha: float(np.quantile(self.train_scores, 1-alpha)) for alpha in alphas}
        self.results = pd.DataFrame({'vae_score':self.validation_scores}, index=self.data.validation.index)
        self.results.index.name = 'source_index'
        for alpha, threshold in self.thresholds.items():
            suffix = f'{int(round(alpha*100)):02d}'
            self.results[f'vae_anomaly_{suffix}'] = (self.validation_scores > threshold).astype(int)
        display(self.results.reset_index(drop=True).head())
        for alpha, threshold in self.thresholds.items():
            count = (self.validation_scores > threshold).sum()
            print(f'Alpha={alpha} | Threshold={threshold:.6f} | Anomalies={count} | Rate={count/len(self.results):.2%}')
        return self

    def save(self):
        # Keep the legacy score CSV layout, with explicit row IDs in a companion file.
        save_table(self.results, self.data.results_dir / 'vae_validation_scores.csv', index=False)
        save_table(pd.DataFrame({'source_index':self.results.index}),
                   self.data.results_dir / 'vae_validation_indices.csv', index=False)
        self.search.save()
        print('All VAE results saved successfully.')
