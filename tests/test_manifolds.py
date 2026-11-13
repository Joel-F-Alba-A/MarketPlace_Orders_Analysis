"""Regression tests for notebook 05's extracted workflows."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.models.manifolds import ManifoldData, TSNESearch, UMAPSearch, read_embedding
from src.evaluation.manifolds import PreservationEvaluation, SeedStability


class ManifoldTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frame = pd.DataFrame(rng.normal(size=(80,3)),columns=['a','b','c'])
        self.frame['Customer Segment'] = ['Consumer']*48+['Corporate']*32
        self.frame['Department Name'] = 'Department'
        self.data = ManifoldData(frame=self.frame,features=['a','b','c']).sample(size=40)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.limits = threadpool_limits(limits=1)
        self.limits.__enter__()

    def tearDown(self):
        self.limits.__exit__(None,None,None)
        self.temporary.cleanup()

    def test_stratified_sample_is_repeatable_and_original_frame_is_unchanged(self):
        other = ManifoldData(frame=self.frame,features=['a','b','c']).sample(size=40)
        pd.testing.assert_frame_equal(self.data.sample_scaled,other.sample_scaled)
        self.assertEqual(self.data.sample_frame['Customer Segment'].value_counts().to_dict(),
                         {'Consumer':24,'Corporate':16})
        pd.testing.assert_frame_equal(self.data.frame,self.frame)
        self.data.sample_frame.iloc[0,0] = 1000
        pd.testing.assert_frame_equal(self.data.frame,self.frame)

    def test_embedding_cache_is_validated_and_force_retrain_is_explicit(self):
        search = TSNESearch(perplexities=[5],results_dir=self.directory)
        path = search.path_for(5)
        embedding = pd.DataFrame(np.random.default_rng(7).normal(size=(40,2)),
                                  index=self.data.sample_scaled.index,columns=['t-SNE 1','t-SNE 2'])
        embedding.to_csv(path,index_label='source_index')
        with patch.object(search,'estimator',side_effect=AssertionError('Cache must load')):
            search.run(self.data)
        embedding.index = embedding.index[::-1]
        embedding.to_csv(path,index_label='source_index')
        with self.assertRaises(ValueError):
            read_embedding(path,self.data.sample_scaled.index,'tsne')
        forced = TSNESearch(perplexities=[5],force_retrain=True,results_dir=self.directory)
        model = Mock()
        model.fit_transform.return_value = np.zeros((40,2))
        with patch.object(forced,'estimator',return_value=model):
            forced.run(self.data)
        model.fit_transform.assert_called_once()

    def test_preservation_matches_independent_metrics_and_resumes_completed_table(self):
        search = TSNESearch(perplexities=[5],results_dir=self.directory)
        values = np.random.default_rng(9).normal(size=(40,2))
        path = search.path_for(5)
        pd.DataFrame(values,index=self.data.sample_scaled.index,
                     columns=['t-SNE 1','t-SNE 2']).to_csv(path,index_label='source_index')
        search.run(self.data)
        evaluation = PreservationEvaluation(ks=[2,5,10],evaluation_size=40).run(search)
        X = self.data.sample_scaled.sample(n=40,random_state=42)
        Y = pd.read_csv(path,index_col='source_index').loc[X.index].to_numpy()
        for k in evaluation.ks:
            self.assertAlmostEqual(evaluation.results.loc[0,f'Trustworthiness@{k}'],
                                   trustworthiness(X.to_numpy(),Y,n_neighbors=k))
            a = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
            b = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False)
            expected = np.mean([len(set(u)&set(v))/k for u,v in zip(a,b)])
            self.assertAlmostEqual(evaluation.results.loc[0,f'Neighborhood Overlap@{k}'],expected)
        with patch('src.evaluation.manifolds.chunked_trustworthiness',
                   side_effect=AssertionError('Completed metrics must not rerun')):
            resumed = PreservationEvaluation(ks=[2,5,10],evaluation_size=40).run(search)
        pd.testing.assert_frame_equal(evaluation.results,resumed.results,check_exact=False)

    def test_umap_multiindex_metrics_and_presentation_reload(self):
        search = UMAPSearch(n_neighbors=(5,), min_dist=(0.1,), results_dir=self.directory)
        values = np.random.default_rng(10).normal(size=(40, 2))
        path = search.path_for((5, 0.1))
        pd.DataFrame(values, index=self.data.sample_scaled.index,
                     columns=['UMAP 1', 'UMAP 2']).to_csv(path, index_label='source_index')
        search.run(self.data)
        evaluation = PreservationEvaluation(ks=[2], evaluation_size=40).run(search)
        self.assertEqual(evaluation.results.loc[0, 'n_neighbors'], 5)
        self.assertEqual(evaluation.results.loc[0, 'min_dist'], 0.1)
        reloaded = PreservationEvaluation.from_csv(evaluation.path)
        pd.testing.assert_frame_equal(evaluation.results, reloaded.results, check_exact=False)
        pd.testing.assert_frame_equal(evaluation.comparison, reloaded.comparison, check_exact=False)

    def test_seed_stability_is_rotation_invariant_and_excludes_diagonal(self):
        stability = SeedStability('tsne',seeds=[0,1],ks=[2,5],results_dir=self.directory)
        values = np.random.default_rng(11).normal(size=(40,2))
        stability.data, stability.run_dir = self.data,self.directory
        stability.embeddings = {0:values,1:values@np.array([[0,-1],[1,0]])+10}
        stability.run()
        for k in stability.ks:
            np.testing.assert_allclose(stability.matrices[k],1)
        self.assertTrue((stability.summary['seed_pairs']==1).all())
        self.assertTrue((stability.summary['mean_overlap']==1).all())


if __name__ == '__main__':
    unittest.main()
