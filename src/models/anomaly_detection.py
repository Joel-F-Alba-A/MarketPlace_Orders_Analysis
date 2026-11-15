"""Training-only preprocessing and anomaly scoring for notebook 06.

Flags use empirical training quantiles, not supervised labels. All public
workflows keep model-specific scores and thresholds on separate objects.
"""
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.config import CLEAN_DATA_PATH, RESULTS_DIR
from src.models.manifolds import FEATURES

NOTEBOOK_RESULTS = RESULTS_DIR / "06_anomaly_detection"
ALPHAS = (0.01, 0.05)


class AnomalyData:
    """The notebook's original 60/20/20 split; scaling is fitted on training only."""

    def __init__(self, path=CLEAN_DATA_PATH, features=FEATURES, frame=None):
        self.frame = pd.read_csv(path) if frame is None else frame.copy(deep=True)
        self.features = list(features)
        self.X = self.frame[self.features].copy()
        if not self.X.index.is_unique or not np.isfinite(self.X.to_numpy(dtype=float)).all():
            raise ValueError("Features need unique row IDs and finite values.")

    def split(self, random_state=42):
        self.train, temporary = train_test_split(self.X, test_size=.40, random_state=random_state)
        self.validation, self.test = train_test_split(temporary, test_size=.50, random_state=random_state)
        return self

    def standardize(self):
        self.scaler = StandardScaler().fit(self.train)
        for split in ("train", "validation", "test"):
            setattr(self, f"{split}_scaled", self.scaler.transform(getattr(self, split)))
        return self


def alpha_suffix(alpha):
    return f"Alpha_{str(alpha).replace('.', '_')}"


class AnomalyDetector:
    """Shared threshold estimation, flags, summaries, and CSV exports."""
    name = ""
    score_columns = {}
    prefix = ""
    result_names = {}

    def __init__(self, alphas=ALPHAS, results_dir=NOTEBOOK_RESULTS):
        self.alphas = list(alphas)
        if not self.alphas or not all(0 < a < 1 for a in self.alphas):
            raise ValueError("Alphas must lie strictly between zero and one.")
        self.results_dir = Path(results_dir)

    def flag_column(self, statistic, alpha):
        if self.name == "LOF":
            return f"Anomaly_alpha_{alpha:.2f}"
        if self.name == "Isolation Forest":
            return f"IF_Anomaly_alpha_{alpha:.2f}"
        if self.name == "Mahalanobis":
            return f"Mahalanobis_Anomaly_{alpha_suffix(alpha)}"
        return f"{statistic}_Anomaly_{alpha_suffix(alpha)}"

    def _finalize(self, data, scores, save=True):
        self.data = data
        self.thresholds = {alpha: {stat: float(np.quantile(scores['train'][stat], 1-alpha))
                                   for stat in self.score_columns} for alpha in self.alphas}
        self.results = {}
        rows = []
        for split, values in scores.items():
            frame = pd.DataFrame({column: values[stat] for stat,column in self.score_columns.items()},
                                  index=getattr(data,split).index.copy())
            frame.index.name = "source_index"
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError("Scores contain nonfinite values.")
            for alpha in self.alphas:
                flags = []
                for stat,column in self.score_columns.items():
                    flag = frame[column] > self.thresholds[alpha][stat]
                    frame[self.flag_column(stat,alpha)] = flag
                    flags.append(flag)
                if len(flags)==2:
                    suffix = alpha_suffix(alpha)
                    frame[f"Either_Anomaly_{suffix}"] = flags[0] | flags[1]
                    frame[f"Both_Anomaly_{suffix}"] = flags[0] & flags[1]
                row = {"Dataset":split.title(),"Alpha":alpha,"Observations":len(frame)}
                for stat,flag in zip(self.score_columns,flags):
                    row[f"{stat} Anomalies"] = int(flag.sum())
                    row[f"{stat} Anomaly Rate (%)"] = 100*flag.mean()
                if len(flags)==2:
                    a,b = flags
                    row.update({"Both":int((a&b).sum()),"Either":int((a|b).sum()),
                                f"{list(self.score_columns)[0]} Only":int((a&~b).sum()),
                                f"{list(self.score_columns)[1]} Only":int((~a&b).sum()),
                                "Either Anomaly Rate (%)":100*(a|b).mean()})
                rows.append(row)
            self.results[split] = frame
        self.summary = pd.DataFrame(rows)
        if len(self.score_columns) == 2:
            first,second = self.score_columns
            if self.name == "PCA":
                self.summary = self.summary[self.summary["Dataset"] != "Train"].reset_index(drop=True)
            else:
                self.summary["Dataset"] = self.summary["Dataset"].replace({"Train":"Training"})
                self.summary["Jaccard"] = self.summary["Both"] / self.summary["Either"].replace(0,np.nan)
            columns = ["Dataset","Alpha",f"{first} Anomalies",f"{second} Anomalies",
                       f"{first} Only",f"{second} Only","Both","Either",
                       f"{first} Anomaly Rate (%)",f"{second} Anomaly Rate (%)","Either Anomaly Rate (%)"]
            if self.name == "ICA":
                columns.append("Jaccard")
            self.summary = self.summary[columns]
        else:
            stat,column = next(iter(self.score_columns.items()))
            self.summary = self.summary.rename(columns={f"{stat} Anomalies":"Anomalies",
                                                        f"{stat} Anomaly Rate (%)":"Anomaly Rate (%)"})
            if self.name == "Mahalanobis":
                self.summary["Dataset"] = self.summary["Dataset"].replace({"Train":"Training"})
            else:
                # Score summaries describe each complete split, as in the original notebook.
                for metric,operation in [("Score Mean",lambda x:x.mean()),("Score Median",lambda x:x.median()),
                                         ("Score P95",lambda x:x.quantile(.95)),("Score P99",lambda x:x.quantile(.99)),
                                         ("Score Max",lambda x:x.max())]:
                    self.summary[metric] = [operation(self.results[row["Dataset"].lower()][column])
                                            for _,row in self.summary.iterrows()]
                if self.name == "Isolation Forest":
                    self.summary.insert(2,"Threshold",[self.thresholds[a][stat] for a in self.summary["Alpha"]])
        threshold_rows = []
        for alpha,thresholds in self.thresholds.items():
            if len(self.score_columns)==2:
                row = {"Alpha":alpha,"Percentile":100*(1-alpha)}
                for stat,value in thresholds.items():
                    row[f"{stat} Threshold"] = value
                for stat,value in thresholds.items():
                    row[f"{stat} Training Anomaly Rate (%)"] = 100*np.mean(scores['train'][stat]>value)
            else:
                stat = next(iter(self.score_columns))
                row = {"Alpha":alpha,"Quantile":1-alpha,"Threshold":thresholds[stat]}
                if self.name == "Mahalanobis":
                    row = {"Alpha":alpha,"Percentile":100*(1-alpha),"Threshold":thresholds[stat],
                           "Training Anomaly Rate (%)":100*np.mean(scores['train'][stat]>thresholds[stat])}
            threshold_rows.append(row)
        self.threshold_table = pd.DataFrame(threshold_rows)
        if save:
            self.results_dir.mkdir(parents=True,exist_ok=True)
            for split,frame in self.results.items():
                frame.to_csv(self.results_dir/self.result_names[split],index_label="source_index")
            self.threshold_table.to_csv(self.results_dir/f"{self.prefix}_thresholds.csv",index=False)
            summary_name = (f"{self.prefix}_detection_summary.csv" if self.name in ("LOF","Isolation Forest")
                            else f"{self.prefix}_comparison.csv")
            self.summary.to_csv(self.results_dir/summary_name,index=False)
        return self

    def display(self):
        from IPython.display import display
        print(self.name)
        display(self.threshold_table.round(6))
        display(self.summary.round(4))
        display(self.results['validation'].head())

    def load_results(self, data):
        """Explicitly load existing experiment tables without refitting models.

        Requires matching split IDs and the threshold-based flags. This is for
        presenting saved results; call fit(data) to recompute after input changes.
        """
        scores = {}
        for split,name in self.result_names.items():
            path = self.results_dir/name
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; run fit(data) first.")
            frame = pd.read_csv(path,index_col="source_index")
            if not frame.index.equals(getattr(data,split).index):
                raise ValueError(f"Saved score rows do not match the {split} split.")
            scores[split] = {stat:frame[column].to_numpy() for stat,column in self.score_columns.items()}
        return self._finalize(data,scores,save=False)


class PCAAnomalyDetector(AnomalyDetector):
    name, prefix = "PCA", "pca_anomaly_quantile"
    score_columns = {"T2":"Hotelling_T2", "SPE":"SPE"}
    result_names = dict(train="pca_anomaly_training.csv",validation="pca_anomaly_validation.csv",test="pca_anomaly_test.csv")

    def __init__(self, n_components=5, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.n_components,self.random_state = n_components,random_state

    def fit(self,data):
        self.model = PCA(n_components=self.n_components,svd_solver="full",random_state=self.random_state)
        with threadpool_limits(limits=1):
            self.model.fit(data.train_scaled)
            scores = {split:self.score(getattr(data,f"{split}_scaled")) for split in ("train","validation","test")}
        print(f"Explained variance: {self.model.explained_variance_ratio_.sum():.4%}")
        return self._finalize(data,scores)

    def score(self,X):
        latent = self.model.transform(X)
        return dict(T2=np.sum(latent**2/self.model.explained_variance_,axis=1),
                    SPE=np.sum((np.asarray(X)-self.model.inverse_transform(latent))**2,axis=1))


class ICAAnomalyDetector(AnomalyDetector):
    name, prefix = "ICA", "ica_anomaly_quantile"
    score_columns = {"NLL":"ICA_NLL","SPE":"ICA_SPE"}
    result_names = dict(train="ica_anomaly_training.csv",validation="ica_anomaly_validation.csv",test="ica_anomaly_test.csv")

    def __init__(self,n_components=5,random_state=42,max_iter=500,tol=1e-3,
                 bandwidth=.3,kde_sample_size=5000,n_jobs=-1,**kwargs):
        super().__init__(**kwargs)
        self.n_components,self.random_state = n_components,random_state
        self.max_iter,self.tol = max_iter,tol
        self.bandwidth,self.kde_sample_size,self.n_jobs = bandwidth,kde_sample_size,n_jobs

    def fit(self,data):
        self.model = FastICA(n_components=self.n_components,algorithm="parallel",whiten="unit-variance",
                             max_iter=self.max_iter,tol=self.tol,random_state=self.random_state)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",ConvergenceWarning)
            self.model.fit(data.train_scaled)
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise RuntimeError("FastICA did not converge; review the configuration.")
        latent = self.model.transform(data.train_scaled)
        indices = np.random.default_rng(self.random_state).choice(
            len(latent),size=min(self.kde_sample_size,len(latent)),replace=False)
        self.kdes = [KernelDensity(kernel="gaussian",bandwidth=self.bandwidth,algorithm="auto",
                                   atol=1e-5,rtol=1e-5).fit(latent[indices,j,None])
                     for j in range(self.n_components)]
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            scores = {split:self.score(getattr(data,f"{split}_scaled")) for split in ("train","validation","test")}
        print(f"ICA converged in {self.model.n_iter_} iterations; scoring: {time.perf_counter()-started:.2f}s")
        return self._finalize(data,scores)

    def score(self,X):
        latent = self.model.transform(X)
        columns = Parallel(n_jobs=self.n_jobs,prefer="threads")(
            delayed(kde.score_samples)(latent[:,j,None]) for j,kde in enumerate(self.kdes))
        return dict(NLL=-np.column_stack(columns).sum(axis=1),
                    SPE=np.sum((np.asarray(X)-self.model.inverse_transform(latent))**2,axis=1))


class MahalanobisDetector(AnomalyDetector):
    name, prefix = "Mahalanobis", "mahalanobis_quantile"
    score_columns = {"D2":"Mahalanobis_D2"}
    result_names = dict(train="mahalanobis_training.csv",validation="mahalanobis_validation.csv",test="mahalanobis_test.csv")

    def fit(self,data):
        self.model = LedoitWolf().fit(np.asarray(data.train_scaled,dtype=np.float64))
        scores = {split:{"D2":self.model.mahalanobis(np.asarray(getattr(data,f"{split}_scaled"),dtype=np.float64))}
                  for split in ("train","validation","test")}
        print(f"Covariance shrinkage: {self.model.shrinkage_:.6f}")
        return self._finalize(data,scores)


class LOFDetector(AnomalyDetector):
    name, prefix = "LOF", "lof_k20"
    score_columns = {"LOF":"LOF_Score"}
    result_names = dict(train="lof_k20_train_scores.csv",validation="lof_k20_validation_scores.csv")

    def __init__(self,n_neighbors=20,n_jobs=-1,**kwargs):
        super().__init__(**kwargs)
        self.n_neighbors,self.n_jobs = n_neighbors,n_jobs
        self.prefix = f"lof_k{n_neighbors}"
        self.result_names = dict(train=f"{self.prefix}_train_scores.csv",validation=f"{self.prefix}_validation_scores.csv")

    def fit(self,data):
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors,novelty=True,
                                       metric="euclidean",algorithm="auto",n_jobs=self.n_jobs).fit(data.train_scaled)
        scores = dict(train={"LOF":-self.model.negative_outlier_factor_},
                      validation={"LOF":-self.model.score_samples(data.validation_scaled)})
        self._finalize(data,scores)
        joblib.dump(self.model,self.results_dir/f"{self.prefix}_model.joblib")
        return self


class IsolationForestDetector(AnomalyDetector):
    name = "Isolation Forest"
    score_columns = {"IF":"IF_Score"}

    def __init__(self,n_estimators=400,max_samples=1024,random_state=42,n_jobs=-1,**kwargs):
        super().__init__(**kwargs)
        self.n_estimators,self.max_samples = n_estimators,max_samples
        self.random_state,self.n_jobs = random_state,n_jobs
        self.prefix = f"if_{n_estimators}trees_{max_samples}samples"
        self.result_names = dict(train=f"{self.prefix}_train_scores.csv",validation=f"{self.prefix}_validation_scores.csv")

    def fit(self,data):
        self.model = IsolationForest(n_estimators=self.n_estimators,max_samples=self.max_samples,
            max_features=1.0,contamination="auto",bootstrap=False,random_state=self.random_state,
            n_jobs=self.n_jobs).fit(data.train_scaled)
        scores = {split:{"IF":-self.model.score_samples(getattr(data,f"{split}_scaled"))}
                  for split in ("train","validation")}
        self._finalize(data,scores)
        joblib.dump(self.model,self.results_dir/f"{self.prefix}_model.joblib")
        return self
