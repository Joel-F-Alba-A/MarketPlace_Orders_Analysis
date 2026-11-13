"""Compatibility helper for the former standalone 2,000-row metric cell."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.manifolds import ManifoldData, TSNESearch
from src.evaluation.manifolds import PreservationEvaluation


def run(data=None):
    data = ManifoldData().sample() if data is None else data
    search = TSNESearch().run(data)
    return PreservationEvaluation(ks=(10,), evaluation_size=min(2000,len(data.sample_scaled))).run(search)


if __name__ == "__main__":
    run().display()
