"""Per-statistic descriptions of flagged/remaining observations, without labels."""
import numpy as np
import pandas as pd
from IPython.display import display


class AnomalyAnalysis:
    """Inspect one detector/statistic/threshold using original numerical features."""

    def __init__(self, detector, statistic, alpha=.05, split="validation"):
        self.detector,self.statistic,self.alpha,self.split = detector,statistic,alpha,split
        self.results = detector.results[split]
        self.threshold = detector.thresholds[alpha][statistic]
        self.score_column = detector.score_columns[statistic]
        self.flag_column = detector.flag_column(statistic,alpha)
        features = getattr(detector.data,split)
        if not self.results.index.equals(features.index):
            raise ValueError("Scores and original features must have matching row IDs and order.")
        self.flags = self.results[self.flag_column].astype(bool)
        self.anomalous,self.normal = features.loc[self.flags].copy(),features.loc[~self.flags].copy()
        self.anomalous_scores = self.results.loc[self.flags,self.score_column]
        self.normal_scores = self.results.loc[~self.flags,self.score_column]
        self.features = pd.concat({"Anomalous":self.anomalous.describe().T,
                                   "Non-Anomalous":self.normal.describe().T},names=["Group","Feature"])
        self.scores = pd.DataFrame({"Anomalous":self.anomalous_scores.describe(),
                                    "Non-Anomalous":self.normal_scores.describe()})
        self.prefix = f"{detector.prefix}_{statistic.lower()}_alpha_{str(alpha).replace('.','_')}"

    def describe(self, save=True):
        print(f"{self.detector.name} {self.statistic} | alpha={self.alpha} | {self.split}")
        print(f"Threshold: {self.threshold:.6f}")
        print(f"Anomalous: {int(self.flags.sum()):,}; non-anomalous: {int((~self.flags).sum()):,}")
        print(f"Anomaly rate: {100*self.flags.mean():.2f}%")
        if self.detector.name=="PCA":
            display(self.anomalous.describe().round(3))
            display(self.normal.describe().round(3))
        else:
            display(self.features.round(4))
        display(self.scores.round(4))
        if save:
            self.detector.results_dir.mkdir(parents=True,exist_ok=True)
            self.features.to_csv(self.detector.results_dir/f"{self.prefix}_features.csv")
            self.scores.to_csv(self.detector.results_dir/f"{self.prefix}_scores.csv")
        return self

    def plot(self, save=True):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(9,6))
        groups = [self.normal_scores.to_numpy(),self.anomalous_scores.to_numpy()]
        ax.boxplot(groups,tick_labels=["Non-Anomalous","Anomalous"],showmeans=True,
                   meanprops=dict(marker="D",markerfacecolor="black",markeredgecolor="black",markersize=7))
        ax.axhline(self.threshold,linestyle="--",label=f"Threshold = {self.threshold:.4f}")
        ax.set(title=f"{self.detector.name} {self.statistic} — alpha={self.alpha}",ylabel=self.score_column)
        ax.grid(axis="y",alpha=.3)
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(self.detector.results_dir/f"{self.prefix}_boxplot.png",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close(fig)
        fig,ax = plt.subplots(figsize=(10,6))
        bins = np.histogram_bin_edges(self.results[self.score_column].dropna(),bins=80)
        ax.hist(self.normal_scores,bins=bins,alpha=.65,label="Non-Anomalous")
        ax.hist(self.anomalous_scores,bins=bins,alpha=.65,label="Anomalous")
        ax.axvline(self.threshold,linestyle="--",label=f"Threshold = {self.threshold:.4f}")
        ax.set(title=f"{self.detector.name} {self.statistic} distribution — alpha={self.alpha}",
               xlabel=self.score_column,ylabel="Frequency")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(self.detector.results_dir/f"{self.prefix}_histogram.png",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def present(self):
        self.describe()
        self.plot()
