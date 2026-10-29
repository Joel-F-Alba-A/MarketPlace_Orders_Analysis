"""Regression checks for the modeling logic extracted from notebook 03."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits

from src.models.dimensionality_reduction import (
    DEFAULT_FEATURES, ReductionData, SVDSearch, PCASearch, PPCASearch,
    SVDAnalysis, PCAAnalysis, PPCAAnalysis, ReductionComparison,
)
from src.models.dimensionality_reduction_plots import plot_search, plot_loadings


class ReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(240, 10)) @ rng.normal(size=(10, 10))
        cls.frame = pd.DataFrame(values, columns=DEFAULT_FEATURES)

    def setUp(self):
        self.data = ReductionData(frame=self.frame)
        self.limits = threadpool_limits(limits=1)
        self.limits.__enter__()

    def tearDown(self):
        plt.close("all")
        self.limits.__exit__(None, None, None)

    def test_split_indices_and_training_only_scaling(self):
        data = self.data
        self.assertEqual(tuple(map(len, (data.train, data.validation, data.test))), (168, 36, 36))
        self.assertFalse(set(data.train.index) & set(data.validation.index))
        self.assertFalse(set(data.train.index) & set(data.test.index))
        self.assertFalse(set(data.validation.index) & set(data.test.index))
        changed = self.frame.copy()
        changed.loc[list(data.validation.index) + list(data.test.index)] += 1000
        other = ReductionData(frame=changed)
        np.testing.assert_allclose(data.train_standardized, other.train_standardized)
        np.testing.assert_allclose(data.train_scaled, other.train_scaled)
        np.testing.assert_allclose(data.scaler.mean_, data.train.mean())
        self.assertFalse(data.svd_scaler.with_mean)

    def test_analysis_matches_original_formulas(self):
        data = self.data
        for analysis_cls, method in [(SVDAnalysis, "svd"), (PCAAnalysis, "pca"),
                                     (PPCAAnalysis, "ppca")]:
            with self.subTest(method=method):
                analysis = analysis_cls(data, k=5).fit()
                train, _, test = data.arrays(method)
                model = (TruncatedSVD(5, algorithm="arpack", random_state=42)
                         if method == "svd" else PCA(5, svd_solver="full"))
                model.fit(train)
                if method == "ppca":
                    W = model.components_.T @ np.diag(np.sqrt(np.maximum(
                        model.explained_variance_ - model.noise_variance_, 0)))
                    latent = np.linalg.solve(W.T @ W + model.noise_variance_ * np.eye(5),
                                             W.T @ (test - model.mean_).T).T
                    reconstructed = latent @ W.T + model.mean_
                    np.testing.assert_allclose(analysis.W, W)
                else:
                    latent = model.transform(test)
                    reconstructed = model.inverse_transform(latent)
                np.testing.assert_allclose(analysis.latent["test"], latent, atol=1e-12)
                np.testing.assert_allclose(analysis.reconstructions["test"], reconstructed)
                self.assertAlmostEqual(analysis.evaluate()["Reconstruction MSE"],
                                       mean_squared_error(test, reconstructed))
                self.assertEqual(list(analysis.latent_frame().index), list(data.train.index))
                self.assertEqual(analysis.loadings.shape, (10, 5))

    def test_search_plots_and_final_comparison(self):
        analyses = []
        for search_cls, analysis_cls in [(SVDSearch, SVDAnalysis),
                                         (PCASearch, PCAAnalysis),
                                         (PPCASearch, PPCAAnalysis)]:
            search = search_cls(k_values=(1, 3, 5), n_bootstrap=2, sample_size=100).run(self.data)
            self.assertEqual(search.results["k"].tolist(), [1, 3, 5])
            self.assertTrue(search.results["Subspace Stability"].between(0, 1 + 1e-12).all())
            self.assertTrue(np.isnan(search.results.loc[0, "Silhouette Score"]))
            train, validation, _ = self.data.arrays(search.method)
            fitted = search.models[5]
            expected = mean_squared_error(validation, fitted.inverse_transform(fitted.transform(validation)))
            self.assertAlmostEqual(search.results.iloc[-1]["Validation MSE"], expected)
            if search.method == "ppca":
                self.assertAlmostEqual(search.results.iloc[-1]["Validation Log-Likelihood"], fitted.score(validation))
            analysis = analysis_cls(self.data, k=5).fit()
            analyses.append(analysis)
            with patch("matplotlib.pyplot.show"):
                plot_search(search)
                plot_loadings(analysis)
            plt.close("all")
        comparison = ReductionComparison(*analyses).run().results
        self.assertEqual(comparison["Model"].tolist(), ["Truncated SVD", "Classical PCA", "PPCA"])
        np.testing.assert_allclose(comparison["Test MSE"],
                                   [a.evaluate()["Reconstruction MSE"] for a in analyses])
        self.assertTrue(comparison.loc[:1, "Avg Log-Likelihood"].isna().all())
        self.assertTrue(np.isfinite(comparison.loc[2, "Avg Log-Likelihood"]))


if __name__ == "__main__":
    unittest.main()
