"""Plot a saved notebook 05 metric table using the shared plotting helper."""
from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsne-matplotlib")
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.manifolds import PreservationEvaluation
from src.models.manifold_plots import plot_preservation as _plot


def plot_preservation(csv_path):
    _plot(PreservationEvaluation.from_csv(csv_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path",type=Path)
    plot_preservation(parser.parse_args().csv_path)
