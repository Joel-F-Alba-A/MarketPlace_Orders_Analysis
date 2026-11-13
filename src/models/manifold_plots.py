"""Presentation helpers for manifold searches, stability, and categorical overlays."""
from math import ceil

import matplotlib.pyplot as plt
import numpy as np


def plot_preservation(evaluation):
    results, ks = evaluation.results, evaluation.ks
    if evaluation.method == "tsne":
        results = results.sort_values("Perplexity")
        fig, axes = plt.subplots(1,3,figsize=(17,5))
        for ax,metric in zip(axes[:2],["Trustworthiness","Neighborhood Overlap"]):
            for k in ks:
                ax.plot(results["Perplexity"],results[f"{metric}@{k}"],marker="o",label=f"k = {k}")
            ax.set_title(metric)
            ax.set_ylabel("Score")
            ax.legend()
        axes[2].plot(results["Perplexity"],results["Global Spearman"],marker="o")
        axes[2].set_title("Global Spearman (independent of k)")
        axes[2].set_ylabel("Rank correlation")
        for ax in axes:
            ax.set_xlabel("Perplexity")
            ax.set_xticks(results["Perplexity"])
            ax.grid(alpha=.3)
        fig.suptitle(f"t-SNE structure preservation — seed {evaluation.random_state}")
        fig.tight_layout()
        for suffix in (".png",".pdf"):
            fig.savefig(evaluation.path.with_suffix(suffix),dpi=180,bbox_inches="tight")
        plt.show()
        return
    results = results.sort_values(["min_dist","n_neighbors"])
    distances = sorted(results["min_dist"].unique())
    neighbors = sorted(results["n_neighbors"].unique())
    for metric,filename in [("Trustworthiness","umap_trustworthiness_sensitivity.png"),
                            ("Neighborhood Overlap","umap_neighborhood_overlap_sensitivity.png")]:
        fig,axes = plt.subplots(ceil(len(ks)/2),2,figsize=(13,13),squeeze=False)
        for ax,k in zip(axes.ravel(),ks):
            for distance in distances:
                subset = results[results["min_dist"]==distance]
                ax.plot(subset["n_neighbors"],subset[f"{metric}@{k}"],marker="o",
                        linewidth=2,label=f"min_dist={distance}")
            ax.set(title=f"{metric}@{k}",xlabel="n_neighbors",ylabel=metric,xticks=neighbors)
            ax.grid(alpha=.3)
            ax.legend()
        for ax in axes.ravel()[len(ks):]:
            fig.delaxes(ax)
        fig.suptitle(f"UMAP Hyperparameter Sensitivity — {metric}",fontsize=15,fontweight="bold")
        fig.tight_layout(rect=[0,0,1,.97])
        fig.savefig(evaluation.path.parent/filename,dpi=300,bbox_inches="tight")
        plt.show()
    fig,ax = plt.subplots(figsize=(9,5))
    for distance in distances:
        subset = results[results["min_dist"]==distance]
        ax.plot(subset["n_neighbors"],subset["Global Spearman"],marker="o",linewidth=2,
                label=f"min_dist={distance}")
    ax.set(title="UMAP Hyperparameter Sensitivity — Global Spearman",xlabel="n_neighbors",
           ylabel="Global Spearman",xticks=neighbors)
    ax.grid(alpha=.3)
    ax.legend(title="UMAP min_dist")
    fig.tight_layout()
    fig.savefig(evaluation.path.parent/"umap_global_spearman_sensitivity.png",dpi=300,bbox_inches="tight")
    plt.show()


def plot_seed_stability(stability):
    fig,axes = plt.subplots(ceil(len(stability.ks)/2),2,figsize=(15,19),
                            constrained_layout=True,squeeze=False)
    flat = axes.ravel()
    for ax,k in zip(flat,stability.ks):
        values = stability.matrices[k].loc[stability.seeds,stability.seeds].to_numpy()
        mean = values[np.triu_indices(len(stability.seeds),1)].mean()
        im = ax.imshow(values,cmap="viridis",vmin=0,vmax=1)
        ax.set_title(f"Neighborhood stability@{k} — mean {mean:.3f}")
        ax.set_xticks(range(len(stability.seeds)),labels=stability.seeds,rotation=45)
        ax.set_yticks(range(len(stability.seeds)),labels=stability.seeds)
        ax.set_xlabel("Random seed")
        ax.set_ylabel("Random seed")
        for i in range(len(values)):
            for j in range(len(values)):
                ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if values[i,j]<.5 else "black")
    for ax in flat[len(stability.ks):]:
        ax.axis("off")
    fig.colorbar(im,ax=flat[:len(stability.ks)].tolist(),
                 label="Mean fraction of shared neighbors",shrink=.65)
    label = "t-SNE" if stability.method=="tsne" else "UMAP"
    fig.suptitle(f"{label} seed stability — {len(stability.data.sample_scaled):,} observations, "
                 f"{len(stability.seeds)} seeds\nConfiguration: {stability.config}")
    for suffix in (".png",".pdf"):
        fig.savefig(stability.run_dir/f"heatmaps{suffix}",dpi=170,bbox_inches="tight")
    plt.show()


def plot_embedding(analysis):
    label = "t-SNE" if analysis.method=="tsne" else "UMAP"
    fig,ax = plt.subplots(figsize=(11,8))
    values = analysis.frame.to_numpy()
    ax.scatter(values[:,0],values[:,1],s=8,alpha=.6,rasterized=True)
    ax.set(title=f"Final {label} Embedding — {analysis.config}",
           xlabel=f"{label} 1",ylabel=f"{label} 2")
    fig.tight_layout()
    plt.show()


def plot_categorical_segments(analysis, top_n=10, plots_per_row=2, rows_per_figure=3):
    """Overlay original categorical labels; never change the original dataframe."""
    label = "t-SNE" if analysis.method=="tsne" else "UMAP"
    data = analysis.data.frame
    embedding = analysis.frame
    categorical = data.select_dtypes(include=["object","string","category"]).columns.tolist()
    print(f"Categorical columns found: {len(categorical)}")
    print(categorical)
    if not data.index.is_unique or not embedding.index.is_unique or not embedding.index.isin(data.index).all():
        raise ValueError("Embedding must reference unique original observation IDs.")
    aligned = data.loc[embedding.index]
    coordinates = embedding.to_numpy()
    per_figure = plots_per_row*rows_per_figure
    for start in range(0,len(categorical),per_figure):
        columns = categorical[start:start+per_figure]
        fig,axes = plt.subplots(ceil(len(columns)/plots_per_row),plots_per_row,
                                figsize=(18,7*ceil(len(columns)/plots_per_row)),
                                squeeze=False,constrained_layout=True)
        for ax,column in zip(axes.ravel(),columns):
            values = aligned[column].astype("string").fillna("Missing").astype(str)
            top = values.value_counts().head(top_n).index.tolist()
            grouped = values.where(values.isin(top),"Other")
            other = grouped.eq("Other").to_numpy()
            if other.any():
                ax.scatter(coordinates[other,0],coordinates[other,1],s=2,alpha=.12,
                           color="lightgray",label=f"Other ({other.sum():,})",rasterized=True)
            cmap = plt.get_cmap("tab10")
            for i,category in enumerate(top):
                mask = grouped.eq(category).to_numpy()
                ax.scatter(coordinates[mask,0],coordinates[mask,1],s=3,alpha=.65,
                           color=cmap(i%10),label=f"{category} ({mask.sum():,})",rasterized=True)
            ax.set(title=f"{label} by {column}",xlabel=f"{label} 1",ylabel=f"{label} 2")
            ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8,markerscale=3,frameon=False)
            ax.set_aspect("equal",adjustable="box")
        for ax in axes.ravel()[len(columns):]:
            ax.axis("off")
        fig.suptitle(f"Marketplace {label} Segmentation — {analysis.config}\n"
                     f"Variables {start+1}–{start+len(columns)}",fontsize=17,fontweight="bold")
        plt.show()
