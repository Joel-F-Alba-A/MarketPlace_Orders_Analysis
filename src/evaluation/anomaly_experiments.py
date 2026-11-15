"""Hyperparameter and seed/subsample experiments extracted from notebook 06.

LOF seeds select training subsets; Isolation Forest seeds randomize the forest.
The scoring conventions and saved table schemas are unchanged.
"""
import time
from itertools import product, combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from IPython.display import display
from src.models.anomaly_detection import NOTEBOOK_RESULTS

class LOFSearch:
    """Notebook 06 LOFSearch experiment with explicit data and configuration."""

    def __init__(self, n_neighbors=(10, 20, 30, 50), n_jobs=-1, top_fraction=0.01, results_dir=NOTEBOOK_RESULTS):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.top_fraction = top_fraction
        self.results_dir = Path(results_dir)

    def run(self, data):
        self.data = data
        X_train_scaled, X_val_scaled = (data.train_scaled, data.validation_scaled)
        X_val = data.validation
        K_VALUES = list(self.n_neighbors)
        N_JOBS = self.n_jobs
        TOP_FRACTION = self.top_fraction
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        X_train_array = np.asarray(X_train_scaled, dtype=np.float64)
        X_val_array = np.asarray(X_val_scaled, dtype=np.float64)
        assert X_train_array.shape[1] == X_val_array.shape[1], 'Training and validation must have the same number of features.'
        assert len(X_train_array) > max(K_VALUES), 'Training dataset must contain more observations than max(K_VALUES).'
        print('=' * 65)
        print('LOF - HYPERPARAMETER EVALUATION')
        print('=' * 65)
        print(f'Training observations: {len(X_train_array):,}')
        print(f'Validation observations: {len(X_val_array):,}')
        print(f'Number of features: {X_train_array.shape[1]}')
        print(f'Neighborhood sizes: {K_VALUES}')

        def top_fraction_indices(scores, fraction=0.01):
            """
            Return the indices of the highest-scoring observations.
            Used only to evaluate ranking agreement.
            """
            scores = np.asarray(scores)
            n_top = max(1, int(np.ceil(len(scores) * fraction)))
            return set(np.argsort(scores)[-n_top:])

        def jaccard_similarity(set_a, set_b):
            """
            Jaccard similarity between two sets.
            """
            union = len(set_a | set_b)
            if union == 0:
                return np.nan
            return len(set_a & set_b) / union
        metric_records = []
        score_store = {}
        runtime_store = {}
        for k in K_VALUES:
            print(f'\nEvaluating LOF with k = {k}')
            start = time.perf_counter()
            lof = LocalOutlierFactor(n_neighbors=k, novelty=True, metric='euclidean', algorithm='auto', n_jobs=N_JOBS)
            lof.fit(X_train_array)
            val_scores = -lof.score_samples(X_val_array)
            elapsed = time.perf_counter() - start
            score_store[k] = val_scores
            runtime_store[k] = elapsed
            metric_records.append({'K': k, 'Score Mean': np.mean(val_scores), 'Score Std': np.std(val_scores), 'Score Median': np.median(val_scores), 'Score Min': np.min(val_scores), 'Score Max': np.max(val_scores), 'Score P95': np.quantile(val_scores, 0.95), 'Score P99': np.quantile(val_scores, 0.99), 'Execution Time (s)': elapsed})
            print(f'Execution time: {elapsed:.2f} seconds')
            print(f'Mean LOF score: {np.mean(val_scores):.4f}')
            print(f'Score standard deviation: {np.std(val_scores):.4f}')
        lof_tuning_metrics = pd.DataFrame(metric_records)
        print('\nLOF - SCORE DISTRIBUTION AND COMPUTATIONAL METRICS')
        display(lof_tuning_metrics.round(4))
        sensitivity_records = []
        for i in range(len(K_VALUES) - 1):
            k1 = K_VALUES[i]
            k2 = K_VALUES[i + 1]
            scores_1 = score_store[k1]
            scores_2 = score_store[k2]
            rho = spearmanr(scores_1, scores_2).statistic
            top_1 = top_fraction_indices(scores_1, TOP_FRACTION)
            top_2 = top_fraction_indices(scores_2, TOP_FRACTION)
            jac = jaccard_similarity(top_1, top_2)
            sensitivity_records.append({'K1': k1, 'K2': k2, 'K Pair': f'{k1} vs {k2}', 'Spearman': rho, 'Top 1% Jaccard': jac})
        lof_sensitivity_df = pd.DataFrame(sensitivity_records)
        print('\nLOF - NEIGHBORHOOD SENSITIVITY')
        display(lof_sensitivity_df.round(4))
        spearman_matrix = pd.DataFrame(np.eye(len(K_VALUES)), index=K_VALUES, columns=K_VALUES)
        jaccard_matrix = spearman_matrix.copy()
        top_sets = {k: top_fraction_indices(score_store[k], TOP_FRACTION) for k in K_VALUES}
        for k1 in K_VALUES:
            for k2 in K_VALUES:
                spearman_matrix.loc[k1, k2] = spearmanr(score_store[k1], score_store[k2]).statistic
                jaccard_matrix.loc[k1, k2] = jaccard_similarity(top_sets[k1], top_sets[k2])
        print('\nSPEARMAN CORRELATION MATRIX')
        display(spearman_matrix.round(4))
        print('\nTOP-1% JACCARD MATRIX')
        display(jaccard_matrix.round(4))
        lof_tuning_metrics.to_csv(output_dir / 'lof_tuning_metrics.csv', index=False)
        lof_sensitivity_df.to_csv(output_dir / 'lof_tuning_sensitivity.csv', index=False)
        spearman_matrix.to_csv(output_dir / 'lof_tuning_spearman_matrix.csv', index_label='K')
        jaccard_matrix.to_csv(output_dir / 'lof_tuning_jaccard_matrix.csv', index_label='K')
        lof_tuning_scores = pd.DataFrame({f'LOF_K_{k}': score_store[k] for k in K_VALUES}, index=X_val.index)
        lof_tuning_scores.to_csv(output_dir / 'lof_tuning_validation_scores.csv', index_label='source_index')
        print('\n' + '=' * 65)
        print('LOF HYPERPARAMETER EVALUATION COMPLETED')
        print('=' * 65)
        print(f'Results saved in: {output_dir}')
        self.lof_tuning_metrics = lof_tuning_metrics
        self.score_store = score_store
        self.lof_sensitivity_df = lof_sensitivity_df
        self.spearman_matrix = spearman_matrix
        self.jaccard_matrix = jaccard_matrix
        return self

class LOFStability:
    """Notebook 06 LOFStability experiment with explicit data and configuration."""

    def __init__(self, n_neighbors=20, seeds=(0, 21, 42, 123, 7, 13, 57, 99, 202, 314), train_fraction=0.8, top_fraction=0.01, results_dir=NOTEBOOK_RESULTS):
        self.n_neighbors = n_neighbors
        self.seeds = seeds
        self.train_fraction = train_fraction
        self.top_fraction = top_fraction
        self.results_dir = Path(results_dir)

    def run(self, data):
        self.data = data
        X_train_scaled, X_val_scaled = (data.train_scaled, data.validation_scaled)
        X_val = data.validation
        K = self.n_neighbors
        SEEDS = list(self.seeds)
        TRAIN_FRACTION = self.train_fraction
        TOP_FRACTION = self.top_fraction
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        X_train_array = np.asarray(X_train_scaled, dtype=np.float64)
        X_val_array = np.asarray(X_val_scaled, dtype=np.float64)
        n_train = len(X_train_array)
        n_subsample = int(n_train * TRAIN_FRACTION)
        assert n_subsample > K
        assert X_train_array.shape[1] == X_val_array.shape[1]

        def get_top_indices(scores, fraction=0.01):
            n_top = max(1, int(np.ceil(len(scores) * fraction)))
            return set(np.argsort(scores)[-n_top:])

        def jaccard_similarity(set_a, set_b):
            union = len(set_a | set_b)
            if union == 0:
                return np.nan
            return len(set_a & set_b) / union
        score_store_seed = {}
        runtime_records = []
        for seed in SEEDS:
            print(f'Evaluating seed {seed}...')
            rng = np.random.default_rng(seed)
            train_indices = rng.choice(n_train, size=n_subsample, replace=False)
            X_train_subset = X_train_array[train_indices]
            start = time.perf_counter()
            lof = LocalOutlierFactor(n_neighbors=K, novelty=True, metric='euclidean', algorithm='auto', n_jobs=-1)
            lof.fit(X_train_subset)
            val_scores = -lof.score_samples(X_val_array)
            elapsed = time.perf_counter() - start
            score_store_seed[seed] = val_scores
            runtime_records.append({'Seed': seed, 'Training Observations': len(X_train_subset), 'Validation Observations': len(X_val_array), 'Execution Time (s)': elapsed, 'Score Mean': np.mean(val_scores), 'Score Median': np.median(val_scores), 'Score Std': np.std(val_scores), 'Score P99': np.quantile(val_scores, 0.99), 'Score Max': np.max(val_scores)})
        lof_seed_metrics = pd.DataFrame(runtime_records)
        print('\nLOF results by seed:')
        display(lof_seed_metrics.round(4))
        top_sets = {seed: get_top_indices(score_store_seed[seed], TOP_FRACTION) for seed in SEEDS}
        pairwise_records = []
        for seed_1, seed_2 in combinations(SEEDS, 2):
            scores_1 = score_store_seed[seed_1]
            scores_2 = score_store_seed[seed_2]
            rho = spearmanr(scores_1, scores_2).statistic
            jaccard = jaccard_similarity(top_sets[seed_1], top_sets[seed_2])
            pairwise_records.append({'Seed 1': seed_1, 'Seed 2': seed_2, 'Seed Pair': f'{seed_1} vs {seed_2}', 'Spearman': rho, 'Top 1% Jaccard': jaccard})
        lof_seed_pairwise = pd.DataFrame(pairwise_records)
        print('\nPairwise seed stability:')
        display(lof_seed_pairwise.round(4))
        stability_summary = pd.DataFrame([{'Metric': 'Spearman', 'Mean': lof_seed_pairwise['Spearman'].mean(), 'Std': lof_seed_pairwise['Spearman'].std(), 'Min': lof_seed_pairwise['Spearman'].min(), 'Max': lof_seed_pairwise['Spearman'].max()}, {'Metric': 'Top 1% Jaccard', 'Mean': lof_seed_pairwise['Top 1% Jaccard'].mean(), 'Std': lof_seed_pairwise['Top 1% Jaccard'].std(), 'Min': lof_seed_pairwise['Top 1% Jaccard'].min(), 'Max': lof_seed_pairwise['Top 1% Jaccard'].max()}])
        print('\nOverall seed stability:')
        display(stability_summary.round(4))
        score_matrix = np.column_stack([score_store_seed[seed] for seed in SEEDS])
        score_mean = np.mean(score_matrix, axis=1)
        score_std = np.std(score_matrix, axis=1)
        score_cv = np.divide(score_std, np.abs(score_mean), out=np.full_like(score_std, np.nan), where=np.abs(score_mean) > 1e-12)
        lof_score_variability = pd.DataFrame({'Mean Score': score_mean, 'Score Std': score_std, 'Score CV': score_cv}, index=X_val.index)
        lof_score_variability.index.name = 'source_index'
        print('\nScore variability across seeds:')
        display(lof_score_variability.describe().round(4))
        spearman_seed_matrix = pd.DataFrame(np.eye(len(SEEDS)), index=SEEDS, columns=SEEDS)
        jaccard_seed_matrix = spearman_seed_matrix.copy()
        for seed_1 in SEEDS:
            for seed_2 in SEEDS:
                if seed_1 == seed_2:
                    continue
                spearman_seed_matrix.loc[seed_1, seed_2] = spearmanr(score_store_seed[seed_1], score_store_seed[seed_2]).statistic
                jaccard_seed_matrix.loc[seed_1, seed_2] = jaccard_similarity(top_sets[seed_1], top_sets[seed_2])
        lof_seed_metrics.to_csv(output_dir / 'lof_k20_seed_metrics.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'), index=False)
        lof_seed_pairwise.to_csv(output_dir / 'lof_k20_seed_pairwise.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'), index=False)
        stability_summary.to_csv(output_dir / 'lof_k20_seed_stability_summary.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'), index=False)
        lof_score_variability.to_csv(output_dir / 'lof_k20_seed_score_variability.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'))
        spearman_seed_matrix.to_csv(output_dir / 'lof_k20_seed_spearman_matrix.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'), index_label='Seed')
        jaccard_seed_matrix.to_csv(output_dir / 'lof_k20_seed_jaccard_matrix.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'), index_label='Seed')
        lof_seed_scores = pd.DataFrame({f'LOF_Seed_{seed}': score_store_seed[seed] for seed in SEEDS}, index=X_val.index)
        lof_seed_scores.index.name = 'source_index'
        lof_seed_scores.to_csv(output_dir / 'lof_k20_seed_validation_scores.csv'.replace('lof_k20', f'lof_k{self.n_neighbors}'))
        print('\nLOF seed stability evaluation completed.')
        self.lof_seed_metrics = lof_seed_metrics
        self.lof_seed_pairwise = lof_seed_pairwise
        self.stability_summary = stability_summary
        self.lof_score_variability = lof_score_variability
        self.spearman_seed_matrix = spearman_seed_matrix
        self.jaccard_seed_matrix = jaccard_seed_matrix
        self.score_store_seed = score_store_seed
        return self

class IsolationForestSearch:
    """Notebook 06 IsolationForestSearch experiment with explicit data and configuration."""

    def __init__(self, n_estimators=(50, 100, 200, 400), max_samples=(256, 512, 1024, 4096, 1.0), random_state=42, top_fraction=0.01, results_dir=NOTEBOOK_RESULTS):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.top_fraction = top_fraction
        self.results_dir = Path(results_dir)

    def run(self, data):
        self.data = data
        X_train_scaled, X_val_scaled = (data.train_scaled, data.validation_scaled)
        X_val = data.validation
        N_ESTIMATORS_VALUES = list(self.n_estimators)
        MAX_SAMPLES_VALUES = list(self.max_samples)
        RANDOM_STATE = self.random_state
        TOP_FRACTION = self.top_fraction
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        X_train_array = np.asarray(X_train_scaled, dtype=np.float64)
        X_val_array = np.asarray(X_val_scaled, dtype=np.float64)
        assert X_train_array.shape[1] == X_val_array.shape[1]
        assert len(X_train_array) >= max(MAX_SAMPLES_VALUES)

        def top_fraction_indices(scores, fraction=0.01):
            n_top = max(1, int(np.ceil(len(scores) * fraction)))
            return set(np.argsort(scores)[-n_top:])

        def jaccard_similarity(set_a, set_b):
            union = len(set_a | set_b)
            return len(set_a & set_b) / union if union else np.nan
        metric_records = []
        score_store_if = {}
        configurations = list(product(N_ESTIMATORS_VALUES, MAX_SAMPLES_VALUES))
        for n_estimators, max_samples in configurations:
            config_name = f'Trees_{n_estimators}_Samples_{max_samples}'
            print(f'Evaluating {config_name}...')
            start = time.perf_counter()
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, max_features=1.0, contamination='auto', bootstrap=False, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X_train_array)
            val_scores = -model.score_samples(X_val_array)
            elapsed = time.perf_counter() - start
            score_store_if[config_name] = val_scores
            metric_records.append({'Configuration': config_name, 'N Estimators': n_estimators, 'Max Samples': max_samples, 'Score Mean': np.mean(val_scores), 'Score Median': np.median(val_scores), 'Score Std': np.std(val_scores), 'Score P95': np.quantile(val_scores, 0.95), 'Score P99': np.quantile(val_scores, 0.99), 'Score Max': np.max(val_scores), 'Execution Time (s)': elapsed})
        if_tuning_metrics = pd.DataFrame(metric_records)
        print('\nISOLATION FOREST HYPERPARAMETER METRICS')
        display(if_tuning_metrics.round(4))
        tree_sensitivity_records = []
        for max_samples in MAX_SAMPLES_VALUES:
            for i in range(len(N_ESTIMATORS_VALUES) - 1):
                n1 = N_ESTIMATORS_VALUES[i]
                n2 = N_ESTIMATORS_VALUES[i + 1]
                key1 = f'Trees_{n1}_Samples_{max_samples}'
                key2 = f'Trees_{n2}_Samples_{max_samples}'
                scores1 = score_store_if[key1]
                scores2 = score_store_if[key2]
                rho = spearmanr(scores1, scores2).statistic
                jaccard = jaccard_similarity(top_fraction_indices(scores1, TOP_FRACTION), top_fraction_indices(scores2, TOP_FRACTION))
                tree_sensitivity_records.append({'Max Samples': max_samples, 'Trees 1': n1, 'Trees 2': n2, 'Comparison': f'{n1} vs {n2}', 'Spearman': rho, 'Top 1% Jaccard': jaccard})
        if_tree_sensitivity = pd.DataFrame(tree_sensitivity_records)
        print('\nSENSITIVITY TO NUMBER OF TREES')
        display(if_tree_sensitivity.round(4))
        sample_sensitivity_records = []
        for n_estimators in N_ESTIMATORS_VALUES:
            for i in range(len(MAX_SAMPLES_VALUES) - 1):
                s1 = MAX_SAMPLES_VALUES[i]
                s2 = MAX_SAMPLES_VALUES[i + 1]
                key1 = f'Trees_{n_estimators}_Samples_{s1}'
                key2 = f'Trees_{n_estimators}_Samples_{s2}'
                scores1 = score_store_if[key1]
                scores2 = score_store_if[key2]
                rho = spearmanr(scores1, scores2).statistic
                jaccard = jaccard_similarity(top_fraction_indices(scores1, TOP_FRACTION), top_fraction_indices(scores2, TOP_FRACTION))
                sample_sensitivity_records.append({'N Estimators': n_estimators, 'Samples 1': s1, 'Samples 2': s2, 'Comparison': f'{s1} vs {s2}', 'Spearman': rho, 'Top 1% Jaccard': jaccard})
        if_sample_sensitivity = pd.DataFrame(sample_sensitivity_records)
        print('\nSENSITIVITY TO SAMPLE SIZE')
        display(if_sample_sensitivity.round(4))
        if_tuning_metrics.to_csv(output_dir / 'if_tuning_metrics.csv', index=False)
        if_tree_sensitivity.to_csv(output_dir / 'if_tree_sensitivity.csv', index=False)
        if_sample_sensitivity.to_csv(output_dir / 'if_sample_sensitivity.csv', index=False)
        if_tuning_scores = pd.DataFrame(score_store_if, index=X_val.index)
        if_tuning_scores.index.name = 'source_index'
        if_tuning_scores.to_csv(output_dir / 'if_tuning_validation_scores.csv')
        print('\nIsolation Forest hyperparameter evaluation completed.')
        self.if_tuning_metrics = if_tuning_metrics
        self.score_store_if = score_store_if
        self.if_tree_sensitivity = if_tree_sensitivity
        self.if_sample_sensitivity = if_sample_sensitivity
        return self

class IsolationForestStability:
    """Notebook 06 IsolationForestStability experiment with explicit data and configuration."""

    def __init__(self, n_estimators=400, max_samples=1024, seeds=(0, 21, 42, 123, 7, 13, 57, 99, 202, 314), top_fraction=0.01, results_dir=NOTEBOOK_RESULTS):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seeds = seeds
        self.top_fraction = top_fraction
        self.results_dir = Path(results_dir)

    def run(self, data):
        self.data = data
        X_train_scaled, X_val_scaled = (data.train_scaled, data.validation_scaled)
        X_val = data.validation
        N_ESTIMATORS = self.n_estimators
        MAX_SAMPLES = self.max_samples
        SEEDS = list(self.seeds)
        TOP_FRACTION = self.top_fraction
        output_dir = self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        X_train_array = np.asarray(X_train_scaled, dtype=np.float64)
        X_val_array = np.asarray(X_val_scaled, dtype=np.float64)

        def get_top_indices(scores, fraction=0.01):
            n_top = max(1, int(np.ceil(len(scores) * fraction)))
            return set(np.argpartition(scores, -n_top)[-n_top:])

        def jaccard_similarity(a, b):
            union = len(a | b)
            return len(a & b) / union if union else np.nan
        seed_scores = {}
        seed_metrics = []
        for seed in SEEDS:
            print(f'Training Isolation Forest with seed={seed}...')
            start = time.perf_counter()
            model = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES, max_features=1.0, contamination='auto', bootstrap=False, random_state=seed, n_jobs=-1)
            model.fit(X_train_array)
            scores = -model.score_samples(X_val_array)
            elapsed = time.perf_counter() - start
            seed_scores[seed] = scores
            seed_metrics.append({'Seed': seed, 'Score Mean': np.mean(scores), 'Score Median': np.median(scores), 'Score Std': np.std(scores), 'Score P95': np.quantile(scores, 0.95), 'Score P99': np.quantile(scores, 0.99), 'Execution Time (s)': elapsed})
        seed_metrics_df = pd.DataFrame(seed_metrics)
        pairwise_records = []
        spearman_matrix = pd.DataFrame(np.eye(len(SEEDS)), index=SEEDS, columns=SEEDS)
        jaccard_matrix = pd.DataFrame(np.eye(len(SEEDS)), index=SEEDS, columns=SEEDS)
        top_sets = {seed: get_top_indices(seed_scores[seed], TOP_FRACTION) for seed in SEEDS}
        for seed_a, seed_b in combinations(SEEDS, 2):
            rho = spearmanr(seed_scores[seed_a], seed_scores[seed_b]).statistic
            jaccard = jaccard_similarity(top_sets[seed_a], top_sets[seed_b])
            spearman_matrix.loc[seed_a, seed_b] = rho
            spearman_matrix.loc[seed_b, seed_a] = rho
            jaccard_matrix.loc[seed_a, seed_b] = jaccard
            jaccard_matrix.loc[seed_b, seed_a] = jaccard
            pairwise_records.append({'Seed A': seed_a, 'Seed B': seed_b, 'Spearman': rho, 'Top 1% Jaccard': jaccard})
        pairwise_df = pd.DataFrame(pairwise_records)
        scores_matrix = np.column_stack([seed_scores[seed] for seed in SEEDS])
        observation_mean = np.mean(scores_matrix, axis=1)
        observation_std = np.std(scores_matrix, axis=1)
        observation_cv = np.divide(observation_std, np.abs(observation_mean), out=np.full_like(observation_std, np.nan), where=np.abs(observation_mean) > 1e-12)
        observation_stability = pd.DataFrame({'Mean Score': observation_mean, 'Score Std': observation_std, 'Score CV': observation_cv}, index=X_val.index)
        observation_stability.index.name = 'source_index'
        print('\n' + '=' * 65)
        print('ISOLATION FOREST - RANDOM SEED STABILITY')
        print('=' * 65)
        print(f'Number of trees: {N_ESTIMATORS}')
        print(f'Samples per tree: {MAX_SAMPLES}')
        print(f'Number of seeds: {len(SEEDS)}')
        print(f'Pairwise comparisons: {len(pairwise_df)}')
        print('\nPAIRWISE STABILITY SUMMARY')
        display(pairwise_df[['Spearman', 'Top 1% Jaccard']].describe().round(4))
        print('\nPER-SEED SCORE METRICS')
        display(seed_metrics_df.round(4))
        print('\nPER-OBSERVATION SCORE STABILITY')
        display(observation_stability.describe().round(4))
        seed_metrics_df.to_csv(output_dir / 'if_seed_metrics.csv', index=False)
        pairwise_df.to_csv(output_dir / 'if_seed_pairwise_stability.csv', index=False)
        spearman_matrix.to_csv(output_dir / 'if_seed_spearman_matrix.csv')
        jaccard_matrix.to_csv(output_dir / 'if_seed_jaccard_matrix.csv')
        observation_stability.to_csv(output_dir / 'if_seed_observation_stability.csv')
        pd.DataFrame(seed_scores, index=X_val.index).to_csv(output_dir / 'if_seed_validation_scores.csv', index_label='source_index')
        print('\nIsolation Forest seed stability analysis completed.')
        self.seed_metrics_df = seed_metrics_df
        self.pairwise_df = pairwise_df
        self.observation_stability = observation_stability
        self.spearman_matrix = spearman_matrix
        self.jaccard_matrix = jaccard_matrix
        self.seed_scores = seed_scores
        return self
