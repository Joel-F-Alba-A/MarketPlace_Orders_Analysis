"""Regression checks for notebook 06's preprocessing, scores, and analyses."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from src.models.anomaly_detection import (
    AnomalyData, PCAAnomalyDetector, ICAAnomalyDetector, MahalanobisDetector,
    LOFDetector, IsolationForestDetector,
)
from src.evaluation.anomaly_analysis import AnomalyAnalysis


class AnomalyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame(rng.laplace(size=(400,6)),columns=list('abcdef'))
        self.data = AnomalyData(frame=self.frame,features=list('abcdef')).split().standardize()
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.limits = threadpool_limits(limits=1)
        self.limits.__enter__()

    def tearDown(self):
        self.limits.__exit__(None,None,None)
        self.temporary.cleanup()

    def test_train_only_scaling_and_original_data_preservation(self):
        self.assertEqual(tuple(len(getattr(self.data,s)) for s in ('train','validation','test')),(240,80,80))
        changed = self.frame.copy()
        changed.loc[self.data.validation.index] += 1000
        changed.loc[self.data.test.index] -= 1000
        other = AnomalyData(frame=changed,features=list('abcdef')).split().standardize()
        np.testing.assert_allclose(self.data.scaler.mean_,other.scaler.mean_)
        np.testing.assert_allclose(self.data.scaler.scale_,other.scaler.scale_)
        pd.testing.assert_frame_equal(self.frame,self.data.frame)
        self.assertFalse(set(self.data.train.index)&set(self.data.validation.index))

    def test_pca_reconstruction_and_t2_match_independent_projection(self):
        detector = PCAAnomalyDetector(n_components=3,results_dir=self.directory).fit(self.data)
        reference = PCA(n_components=3,svd_solver='full').fit(self.data.train_scaled)
        values = self.data.validation_scaled
        centered = values-reference.mean_
        projection = reference.components_.T@reference.components_
        residual = centered-centered@projection
        expected = np.sum(residual**2,axis=1)
        np.testing.assert_allclose(detector.results['validation']['SPE'],expected,atol=1e-12)
        self.assertEqual(detector.results['validation'].index.tolist(),self.data.validation.index.tolist())

    def test_ica_density_score_and_residuals(self):
        detector = ICAAnomalyDetector(n_components=3,kde_sample_size=100,n_jobs=1,
                                      max_iter=3000,results_dir=self.directory).fit(self.data)
        latent = detector.model.transform(self.data.validation_scaled)
        # Gaussian KDE density computed directly, independently of score_samples.
        bandwidth = detector.bandwidth
        reference=[]
        for j,kde in enumerate(detector.kdes):
            training_values = np.asarray(kde.tree_.data).ravel()
            density = norm.pdf((latent[:,j,None]-training_values)/bandwidth).mean(axis=1)/bandwidth
            reference.append(-np.log(density))
        np.testing.assert_allclose(detector.results['validation']['ICA_NLL'],
                                   np.sum(reference,axis=0),rtol=1e-4,atol=1e-3)
        pca=PCA(n_components=3,svd_solver='full').fit(self.data.train_scaled)
        residual=self.data.validation_scaled-pca.inverse_transform(pca.transform(self.data.validation_scaled))
        np.testing.assert_allclose(detector.results['validation']['ICA_SPE'],(residual**2).sum(axis=1),atol=1e-10)

    def test_flags_are_nested_training_thresholds_and_scores_survive_reload(self):
        for detector in [MahalanobisDetector(results_dir=self.directory),
                         LOFDetector(n_neighbors=10,n_jobs=1,results_dir=self.directory),
                         IsolationForestDetector(n_estimators=10,max_samples=64,n_jobs=1,results_dir=self.directory)]:
            detector.fit(self.data)
            stat = next(iter(detector.score_columns))
            flags = detector.results['validation']
            self.assertTrue((~flags[detector.flag_column(stat,.01)]|flags[detector.flag_column(stat,.05)]).all())
            self.assertGreater(detector.thresholds[.01][stat],detector.thresholds[.05][stat])
            old = {split:frame.copy() for split,frame in detector.results.items()}
            detector.load_results(self.data)
            for split,frame in old.items():
                pd.testing.assert_frame_equal(frame,detector.results[split],check_exact=False,atol=1e-10)

    def test_analysis_uses_original_units_and_detector_specific_threshold(self):
        detector=PCAAnomalyDetector(n_components=3,results_dir=self.directory).fit(self.data)
        original=self.data.frame.copy()
        analysis=AnomalyAnalysis(detector,'SPE',alpha=.05)
        pd.testing.assert_frame_equal(analysis.anomalous,self.data.validation.loc[analysis.flags])
        self.assertEqual(analysis.threshold,detector.thresholds[.05]['SPE'])
        self.assertEqual(len(analysis.anomalous)+len(analysis.normal),len(self.data.validation))
        pd.testing.assert_frame_equal(self.data.frame,original)


if __name__=='__main__':
    unittest.main()
