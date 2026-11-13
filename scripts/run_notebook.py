"""Execute a notebook with live terminal output and save an executed copy."""
import argparse
import os
from pathlib import Path
import sys

# Set before starting the kernel; constrain native libraries in all cells.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/project_final_matplotlib")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


class ConsoleClient(NotebookClient):
    def create_kernel_manager(self):
        manager = super().create_kernel_manager()
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
        ]
        return manager

    def process_message(self, msg, cell, cell_index):
        content = msg.get("content", {})
        kind = msg.get("msg_type", msg.get("header", {}).get("msg_type"))
        if kind == "stream":
            target = sys.stderr if content.get("name") == "stderr" else sys.stdout
            print(content.get("text", ""), end="", file=target, flush=True)
        elif kind in ("execute_result", "display_data"):
            text = content.get("data", {}).get("text/plain")
            if text:
                print(text, flush=True)
        elif kind == "error":
            print("\n".join(content.get("traceback", [])), file=sys.stderr, flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="notebooks/05_manifolds_tsne_umap_embeddings.ipynb")
    args = parser.parse_args()
    path = Path(args.notebook)
    if not path.is_absolute():
        path = ROOT / path
    notebook = nbformat.read(path, as_version=4)
    # Use the exact Python interpreter running this script, not an unrelated
    # kernel named in the notebook's metadata.
    output_dir = ROOT / "results" / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{path.stem}_executed.ipynb"
    client = ConsoleClient(notebook, timeout=None, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}})
    try:
        print(f"Executing all cells: {path}\nPython: {sys.executable}", flush=True)
        client.execute()
    finally:
        nbformat.write(notebook, output)
        print(f"\nExecuted notebook (including partial progress): {output}", flush=True)


if __name__ == "__main__":
    main()
