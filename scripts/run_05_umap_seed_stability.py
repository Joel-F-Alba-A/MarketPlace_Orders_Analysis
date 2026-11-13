"""Run the shared UMAP seed experiment; --force-retrain creates a fresh run."""
from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/umap-stability-mpl")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.manifolds import ManifoldData, NOTEBOOK_RESULTS
from src.evaluation.manifolds import SeedStability
from src.models.manifold_plots import plot_seed_stability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-retrain",action="store_true")
    args = parser.parse_args()
    data = ManifoldData().sample()
    stability = SeedStability("umap",force_retrain=args.force_retrain).fit(data).run()
    print(stability.summary.to_string(index=False),flush=True)
    plot_seed_stability(stability)
    (NOTEBOOK_RESULTS/"umap_stability_latest_run.txt").write_text(str(stability.run_dir))
    print("DONE",flush=True)


if __name__ == "__main__":
    main()
