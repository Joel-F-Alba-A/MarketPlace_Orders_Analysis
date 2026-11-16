# DataCo Order Analytics

A Python project for exploring transaction profiles, latent commercial patterns,
customer–product relationships, and unusual orders in supply-chain data.

The project combines reusable analysis code in `src/` with notebooks for inspection
and interpretation. A single orchestrator runs the analysis pipeline and saves
executed notebook copies, while individual modules expose data preparation,
modeling, evaluation, and presentation methods.

## Capabilities

| Area | Implemented methods | Main outputs |
|---|---|---|
| Data preparation | Missingness, duplicate, constant-column and redundancy checks; month extraction | Processed dataset, feature dictionaries, before/after quality tables |
| Transaction profiling | Gaussian Mixture Models fitted with EM | Probabilistic profiles, membership uncertainty, initialization stability |
| Linear representations | Truncated SVD, PCA, PPCA, Factor Analysis, ICA | Latent coordinates, loadings, reconstruction metrics, bootstrap stability |
| Recommendation | Customer–product matrix, SVD recommender, popularity baseline | Top-K evaluation, product factors, popularity concentration |
| Nonlinear visualization | t-SNE, UMAP | 2D embeddings, neighborhood preservation, distance-rank correlation, categorical overlays |
| Anomaly detection | PCA T²/SPE, ICA NLL/SPE, Mahalanobis, LOF, Isolation Forest, VAE | Scores, training-derived thresholds, flagged observations, descriptive comparisons |
| Neural representations | Autoencoder and variational autoencoder | Checkpoints, reconstruction metrics, latent interpretation, seed stability |
| Product associations | Apriori | Frequent itemsets, support, confidence, lift, readable product rules |

LSA is discussed but not implemented. FP-Growth is a possible future extension; the current
association-rule implementation uses Apriori.

## Quick start

Run the following from this directory, which contains `src/`, `notebooks/`, and
`requirements.txt`. The project has been exercised with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dataco-analytics --display-name "DataCo Analytics"
jupyter lab notebooks/main.ipynb
```

Select **DataCo Analytics** as the notebook kernel, then choose **Run All**.
Provide the raw data described below before starting the pipeline.

Dependencies are listed in [requirements.txt](requirements.txt). They are not
version-pinned. The environment inventory in
[the project overview](notebooks/01_problem_data_reproducibility.ipynb) records
installed versions when executed. PyTorch uses a GPU when available; GPU support
also depends on the installed PyTorch build and the host drivers. CPU execution
is supported.

## Data

Place these files under `data/raw/`:

```text
data/raw/DataCoSupplyChainDataset.csv
data/raw/DescriptionDataCoSupplyChain.csv
```

The first file is the DataCo supply-chain dataset; the second supplies feature
definitions. Raw CSVs are read using Latin-1 encoding. Data files are excluded
from version control and must be supplied separately.

The cleaning stage writes:

```text
data/processed/DataCoChain.csv
```

For the dataset used here, preprocessing retains **180,519 records** and reduces
**53 original columns to 33**. It removes incomplete, constant, redundant, and
out-of-scope fields and derives the order month. It does not remove rows or
blanket-filter loss-making transactions. Three missing customer postal codes
remain outside the numerical model inputs.

A row represents an order item, rather than necessarily a unique order. Customer
and product identifiers are retained for interaction analysis. Association rules
use **customer purchase histories**, not individual order baskets; quantities are
converted to Boolean purchase presence for mining itemsets.

Paths and the reference seed are centralized in [src/config.py](src/config.py).
The processed CSV stays in original units; model-specific preparation applies
scaling separately.

## Run the pipeline

[main.ipynb](notebooks/main.ipynb) runs the following sequence by default:

```text
preprocessing → project overview → GMM → dimensionality reduction
→ factor analysis and recommendation → embeddings → anomaly detection
→ AE/VAE → association rules
```

Each child notebook runs in an independent kernel using main's Python environment.
Variables do not carry over between notebooks. This order matters: preprocessing
creates the processed data, and the factor comparison consumes results exported
by dimensionality reduction.

Main exposes three configuration values:

- `RUN_PREPROCESSING`: defaults to `True`. Set it to `False` only when the processed
  dataset is available and current.
- `NOTEBOOKS`: ordered list of analyses to execute.
- `CELL_TIMEOUT_SECONDS`: defaults to `None` to allow long-running training cells.

Progress reports completed cells and successful notebook completion. A failure
stops the sequence and preserves partial outputs. Each run creates:

```text
results/pipeline_runs/<timestamp>/
├── <executed notebook copies>.ipynb
└── status.json
```

Source notebooks are not overwritten. Preprocessing does rewrite the processed
CSV, and individual analyses write to their configured results directories.

### Run from a terminal

The console runner uses the active Python interpreter, streams cell output, and
saves an executed copy even if a cell fails:

```bash
# Run the entire configured pipeline.
python scripts/run_notebook.py notebooks/main.ipynb

# Run one analysis after preparing its data and prerequisite outputs.
python scripts/run_notebook.py notebooks/06_anomaly_detection.ipynb
```

The runner saves its copy under `results/<notebook stem>/`. When running main,
its child outputs additionally appear under `results/pipeline_runs/`.

A fresh execution can involve multiple searches, repeated seed fits, and expensive
embedding metrics. CPU, GPU, memory, and runtime needs depend on which stages and
cached artifacts are available.

## Explore the analyses

| Notebook | Purpose |
|---|---|
| [00 — Cleaning](notebooks/00_cleaning_and_preprocessing.ipynb) | Prepare data and inspect quality changes |
| [01 — Overview](notebooks/01_problem_data_reproducibility.ipynb) | Data dictionary, feature usage, environment, reproducibility |
| [02 — GMM](notebooks/02_gmm_latent_variables_em.ipynb) | Transaction profiles and probabilistic assignments |
| [03 — Reduction](notebooks/03_svd_pca_probabilistic_pca_lsa.ipynb) | SVD, PCA, PPCA and reconstruction comparisons |
| [04 — Factors and recommendation](notebooks/04_recommendation_ica_matrix_factorization.ipynb) | FA, ICA and customer–product recommendation |
| [05 — Embeddings](notebooks/05_manifolds_tsne_umap_embeddings.ipynb) | t-SNE/UMAP geometry, stability and interpretation |
| [06 — Anomalies](notebooks/06_anomaly_detection.ipynb) | Complementary numerical anomaly scores and thresholds |
| [07 — Neural representations](notebooks/07_neural_networks_autoencoders_vae.ipynb) | AE/VAE selection, latent dimensions and anomalies |
| [08 — Associations](notebooks/08_support_confidence_lift_and_rules.ipynb) | Product itemsets and association rules |
| [09 — Findings](notebooks/09_association_rules_course_integration.ipynb) | Results synthesis, limitations and next steps |

Notebook 09 is a narrative summary with embedded figures and is not part of main's
execution list. Its interpretations should be reviewed when data or configurations
change.

## Architecture

```text
src/
├── config.py                 Shared paths and reference seed
├── preprocessing/            Cleaning, EDA, descriptions and environment reports
├── models/                   Data/model classes and task-specific plotting helpers
└── evaluation/               Preservation, stability and anomaly analysis
notebooks/                    Orchestration and analysis presentation
scripts/                      Console execution and focused experiment utilities
tests/                        Automated model and evaluation checks
data/raw/                     External source files
data/processed/               Prepared analytical dataset
results/                      Tables, checkpoints, experiment metadata and executed runs
output/pdf/                   Executive report, LaTeX source and exported figures
requirements.txt              Explicit project dependencies
```

For example, the association workflow can be used directly from Python when
running from the project root:

```python
from src.models.association_rules import AssociationData, AprioriRules

data = AssociationData()
data.quantity_preview()
data.binary_preview()

analysis = AprioriRules(data, min_support=0.05, min_confidence=0.50)
analysis.fit_itemsets()
analysis.generate_rules()
analysis.summarize()
positive_rules = analysis.positive()
readable_rules = analysis.readable(use_names=True)
```

Notebooks use these classes to keep analytical logic separate from presentation.
After editing imported modules, restart the notebook kernel before rerunning.

## Results and reproducibility

- **Reference seed:** 42. Stability experiments use multiple controlled seeds;
  LOF varies reference samples rather than a model initialization.
- **Scaling:** fitted on training data in held-out analyses. SVD scales without
  centering. Exploratory t-SNE/UMAP standardize the full selected dataset before
  sampling; their geometry metrics are not predictive test accuracy.
- **Caching:** reuse is specific to each implementation. Several searches validate
  saved data/configuration metadata; others fit again. Existing results do not
  guarantee that every stage will skip computation.
- **Refitting:** supported search/stability classes expose `force_retrain` options.
  EDA reports expose `force_refresh` and `use_cache`. Inspect the relevant class
  before changing these options; refitting can replace or create artifacts.
- **Portability:** seeds alone do not guarantee identical floating-point results
  across dependency versions, hardware, or parallel execution. A fresh checkout
  may not include local data or binary checkpoints.

Task outputs live under `results/<analysis>/`. The human-readable synthesis is in
[notebook 09](notebooks/09_association_rules_course_integration.ipynb), with an
[executive report](output/pdf/informe_ejecutivo.pdf) and editable
[LaTeX source](output/pdf/informe_ejecutivo.tex). The [English report](output/pdf/executive_report_en.pdf) and its
[LaTeX source](output/pdf/executive_report_en.tex) are maintained separately from
the Spanish version.

## Tests

From the project root, using the same environment as the notebooks:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The suite covers dimensionality reduction, manifold metrics, anomaly detection,
neural training/evaluation behavior, and association-rule aggregation and metrics. It does not replace a full pipeline run
or verify every narrative interpretation.

## Interpretation boundaries

Anomaly scores identify statistical atypicality, not confirmed fraud, errors, or
business harm. Thresholds use training-score percentiles; without verified labels,
5% and 1% criteria are not measured false-positive rates. Profitable and
loss-making transactions can both be flagged.

Latent factors and 2D maps support descriptive interpretation rather than causal
claims. Their structure depends on selected features, preprocessing, parameters,
and seeds. Low reconstruction error does not by itself establish stable or
commercially useful factors.

Recommendation and association results are influenced by product popularity and
customer activity. Confidence and lift describe observed co-purchases across
histories; they do not establish purchase sequence or the incremental effect of a
promotion. FP-Growth, temporal validation, broader recommendation evaluation, and
business review of individual anomalies remain useful next steps.

## Script entry points and generated artifacts

| Script | Purpose |
|---|---|
| `run_notebook.py` | Execute one notebook or main with live terminal output |
| `05_preservation_metrics_multi_k.py` | Run the shared t-SNE/UMAP preservation workflow |
| `run_05_umap_seed_stability.py` | Run or resume the shared UMAP seed experiment |
| `plot_05_preservation.py` | Plot a saved preservation metric table |
| `tsne_preservation_cell.py` | Compatibility entry point for the earlier 2,000-row evaluation |
| `build_executive_figures.py` | Rebuild the four-panel figure gallery from saved results |
| `build_report_figures.py` | Rebuild the report's reconstruction and anomaly charts (`--language es` or `en`) |

For example: `python scripts/build_report_figures.py --language en` regenerates
English charts without fitting models. Report sources use relative figure paths;
compile each `.tex` from `output/pdf/` with a LaTeX installation supporting the
packages declared in its preamble.

CSV/JSON experiment summaries, selected figures, notebook outputs, and report
PDFs document completed runs. Cached reports may have multiple fingerprints for
different inputs or settings. Local model weights, numeric checkpoint arrays,
executed-run copies, temporary inspection renders, and LaTeX intermediates are
excluded from new commits. They remain available locally where present; missing
weights may require refitting even when summary CSVs are available. The `tmp/`
folder is disposable local inspection output, not an input to the pipeline.
