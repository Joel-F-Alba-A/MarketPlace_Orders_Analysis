"""Notebook 07 regression checks: leakage, checkpoint reuse, scores and row IDs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.models.neural_networks import (
    NeuralData, Autoencoder, VAE, train_ae, train_vae, vae_loss, vae_scores,
    AESearch, VAESearch, VAEAnalysis, AETrainer, save_checkpoint,
)
from src.evaluation.ae_stability import AESeedStability
from src.evaluation.neural_networks import VAEAnomalyAnalysis, VAESeedStability


class NeuralTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.frame = pd.DataFrame(np.random.default_rng(8).normal(size=(200,4)), columns=list('abcd'))
        self.frame.index = np.arange(1000,1200)
        self.data = NeuralData(frame=self.frame, features=list('abcd'),
            device='cpu', results_dir=self.directory).prepare()

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        self.temporary.cleanup()

    def test_scaling_uses_training_only_and_preserves_rows(self):
        changed = self.frame.copy()
        changed.loc[self.data.validation.index] += 1000
        changed.loc[self.data.test.index] -= 1000
        other = NeuralData(frame=changed,features=list('abcd'),device='cpu',
                           results_dir=self.directory).prepare()
        np.testing.assert_array_equal(self.data.scaler.mean_,other.scaler.mean_)
        np.testing.assert_array_equal(self.data.scaler.scale_,other.scaler.scale_)
        self.assertEqual([len(getattr(self.data,s)) for s in ('train','validation','test')],[120,40,40])
        self.assertFalse(set(self.data.train.index) & set(self.data.validation.index))
        pd.testing.assert_frame_equal(self.frame,self.data.frame)

    def test_vae_loss_has_correct_gaussian_kl_and_feature_mean_scaling(self):
        x = torch.tensor([[2.,4.,6.]])
        reconstruction = torch.tensor([[2.2,3.8,5.5]])
        mu,logvar = torch.tensor([[.5]]),torch.zeros((1,1))
        loss,mse,kl = vae_loss(x,reconstruction,mu,logvar,beta=.01)
        self.assertAlmostEqual(mse.item(),.11,places=6)
        self.assertAlmostEqual(kl.item(),.125,places=6)
        self.assertAlmostEqual(loss.item(),.11125,places=6)

    def test_trainers_restore_the_best_validation_epoch(self):
        model,metrics,history = train_ae(self.data,latent_dim=2,hidden_dim=8,
            batch_size=16,seed=42,max_epochs=3,patience=2,show_progress=False)
        with torch.no_grad():
            mse = (model(self.data.validation_tensor)-self.data.validation_tensor).square().mean().item()
        self.assertAlmostEqual(mse,metrics['val_mse'],places=6)
        self.assertAlmostEqual(history.loc[history.epoch.eq(metrics['best_epoch']),'val_mse'].iloc[0],mse,places=6)
        config=dict(latent_dim=2,hidden_dim=8,beta=.01,learning_rate=.001)
        vae,metrics,history = train_vae(self.data,config,seed=42,max_epochs=3,patience=2,
                                       batch_size=16,independent_shuffle=True)
        self.assertEqual(history[metrics['epoch']-1],metrics)
        with torch.no_grad():
            x=self.data.validation_tensor
            hidden=vae.encoder(x)
            mu,logvar=vae.fc_mu(hidden),vae.fc_logvar(hidden)
            loss,_,_=vae_loss(x,vae.decoder(mu),mu,logvar,beta=config['beta'])
        self.assertAlmostEqual(loss.item(),metrics['val_loss'],places=6)

    def test_cached_searches_never_train_and_reject_stale_dataset(self):
        ae_config=dict(latent_dim=2,hidden_dim=8,learning_rate=.001,batch_size=16)
        pd.DataFrame([{**ae_config,'train_mse':.1,'val_mse':.2,'seed':42,
            'best_epoch':1,'epochs_trained':1,'dataset_sha256':self.data.dataset_sha256}]).to_csv(
                self.directory/'ae_hyperparameter_search.csv',index=False)
        with patch('src.models.neural_networks.train_ae',side_effect=AssertionError('unexpected fit')):
            result=AESearch(grid={k:[v] for k,v in ae_config.items()}).run(self.data)
        self.assertEqual(len(result.results),1)
        config=dict(latent_dim=2,hidden_dim=8,beta=.01,learning_rate=.001)
        pd.DataFrame([{**config,'best_epoch':1,'val_loss':.2}]).to_csv(
            self.directory/'vae_hyperparameter_search.csv',index=False)
        pd.DataFrame([dict(epoch=1,train_loss=.3,val_loss=.2)]).to_csv(
            self.directory/'vae_training_history.csv',index=False)
        model=VAE(4,2,8)
        save_checkpoint(dict(model_state_dict=model.state_dict(),config=config,input_dim=4,
            data_fingerprint=self.data.fingerprint),self.directory/'vae_best_model.pt')
        search=VAESearch(grid={k:[v] for k,v in config.items()})
        with patch('src.models.neural_networks.train_vae',side_effect=AssertionError('unexpected fit')):
            search.run(self.data)
        self.assertEqual(search.config,config)
        other=NeuralData(frame=self.frame+1,features=list('abcd'),device='cpu',
                         results_dir=self.directory).prepare()
        with self.assertRaisesRegex(ValueError,'data mismatch'):
            search.run(other)

    def test_posterior_mean_scores_flags_and_original_units(self):
        model=VAE(4,2,8)
        # Large decoder noise in forward() must not affect deterministic scoring.
        first=vae_scores(model,self.data.validation_array)
        second=vae_scores(model,self.data.validation_array)
        np.testing.assert_array_equal(first,second)
        search=type('Search',(),dict(data=self.data,model=model))()
        analysis=VAEAnalysis(search).score().flag()
        np.testing.assert_array_equal(analysis.validation_scores,first)
        self.assertEqual(analysis.results.index.tolist(),self.data.validation.index.tolist())
        self.assertTrue((analysis.results.vae_anomaly_01<=analysis.results.vae_anomaly_05).all())
        original=VAEAnomalyAnalysis(analysis,.05)
        pd.testing.assert_frame_equal(original.anomalous,self.data.validation.loc[original.flags])
        pd.testing.assert_frame_equal(self.frame,self.data.frame)

    def test_ae_stability_accepts_a_bound_trainer_without_notebook_globals(self):
        experiment=AESeedStability.__new__(AESeedStability)
        experiment.train=self.data.train_array
        experiment.validation=self.data.validation_array
        experiment._check_training_inputs(AETrainer(self.data).train)
        wrong=NeuralData(frame=self.frame+1,features=list('abcd'),device='cpu',
                        results_dir=self.directory).prepare()
        wrong.train_tensor=wrong.train_tensor+10
        with self.assertRaises(AssertionError):
            experiment._check_training_inputs(AETrainer(wrong).train)

    def test_stability_rejects_duplicate_or_incomplete_pairs(self):
        experiment=VAESeedStability(seeds=[0,21,42])
        experiment.pairwise=pd.DataFrame([dict(seed_a=0,seed_b=21,spearman=.5,jaccard_05=.3,jaccard_01=.2)]*3)
        with self.assertRaisesRegex(ValueError,'Incomplete or duplicated'):
            experiment._validate()


if __name__=='__main__':
    unittest.main()
