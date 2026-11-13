"""Run notebook 05's shared preservation workflow from the console."""
from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsne-matplotlib")
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.manifolds import ManifoldData, TSNESearch, UMAPSearch
from src.evaluation.manifolds import PreservationEvaluation
from src.models.manifold_plots import plot_preservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=["tsne","umap"],default="tsne")
    args = parser.parse_args()
    data = ManifoldData().sample()
    search = (TSNESearch() if args.method=="tsne" else UMAPSearch()).run(data)
    evaluation = PreservationEvaluation().run(search)
    print(evaluation.results.round(6).to_string(index=False),flush=True)
    plot_preservation(evaluation)
    print("DONE: all metrics and plots saved.",flush=True)


if __name__ == "__main__":
    main()
