# Unsupervised Learning Final Project

## Structure

```text
notebooks/        One preparation notebook and one notebook per rubric criterion
src/              Reusable Python code imported by the notebooks
  config.py       Shared paths and random seed
  preprocessing/  Cleaning, feature selection, and data splitting
    cleaning.py
    splitting.py
  models/         Reusable modeling code
    mixtures.py
    dimensionality_reduction.py
    anomaly_detection.py
    neural_networks.py
  evaluation/
    metrics.py    Metrics and validation functions
  visualization.py Plotting functions
data/             Cleaned data and derived datasets
results/          Figures, tables, and final executive report
requirements.txt  Initial dependencies
```

The original CSV, assignment PDF, downloads, and existing `main.ipynb` are preserved in their original locations. Their original names and contents are unchanged.

## Notebooks

- `00_cleaning_and_feature_selection.ipynb`: Data cleaning and feature selection.
- `01_problem_data_reproducibility.ipynb`: Problem definition, data, and reproducibility (10%).
- `02_gmm_latent_variables_em.ipynb`: Gaussian mixture models, latent variables, and EM (15%).
- `03_svd_pca_probabilistic_pca_lsa.ipynb`: SVD, classical PCA, probabilistic PCA, and LSA (15%).
- `04_recommendation_ica_matrix_factorization.ipynb`: Recommendation systems, ICA, and matrix factorization (15%).
- `05_manifolds_tsne_umap_embeddings.ipynb`: Manifold hypothesis, t-SNE, UMAP, and embedding validation (15%).
- `06_anomaly_detection.ipynb`: Anomaly detection: PCA/ICA, statistical methods, Isolation Forest, and LOF (15%).
- `07_neural_networks_autoencoders_vae.ipynb`: Neural networks, perceptron, backpropagation, optimization, autoencoders, and VAE (10%).
- `08_association_rules_course_integration.ipynb`: Association rules and course integration (5%).

Start with notebook 00 and save prepared data to `data/cleaned_data.csv`. The other notebooks use that path. These notebooks are templates; no analysis has been executed. Choose transformations appropriate to each method and avoid information leakage.

Compare at least two relevant approaches: a baseline and a more complex alternative. Justify which techniques apply to the dataset. Save outputs in `results/`; the executive report must be a PDF of at most four pages, excluding references. Prepare for the written defense in pairs.

## Reusable code

Each package contains an `__init__.py` file. The new modules are placeholders for your own functions and classes; no training or splitting logic has been implemented.

Write functions in `src/` and import them after the setup cell in any notebook:

```python
# In src/preprocessing/cleaning.py:
def select_features(frame, columns):
    return frame.loc[:, columns].copy()
```

```python
# In a notebook, after running its setup cell:
from src.preprocessing.cleaning import select_features

# selected = select_features(df, ["feature_a", "feature_b"])
```

Every notebook includes setup for importing `src/` from either the project root or the `notebooks/` folder. Shared paths in `src/config.py` resolve relative to the project, independent of the notebook location. Restart the kernel after changing imported modules, or enable IPython autoreload during development:

```python
%load_ext autoreload
%autoreload 2
```

## Run

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Select this environment as the notebook kernel. Use explicit random seeds and record package versions once the environment is validated. Additional dependencies depend on the selected algorithms.


### Notebook 00 helpers

`src/preprocessing/eda.py` provides `DatasetEDA`, which displays EDA results and returns summary tables where applicable:

```python
from src.preprocessing.eda import DatasetEDA

eda = DatasetEDA(df)
missing_values = eda.missing_values()
identical_columns = eda.identical_columns()
numerical_stats = eda.numerical_statistics()
```

`src/preprocessing/cleaning.py` provides `add_order_month`, which returns a copy of the dataframe. `src/preprocessing/descriptions.py` provides `show_column_descriptions`, including custom description overrides. Notebook 00 keeps feature-removal choices and observations visible, and saves its processed CSV to `data/processed/DataCoChain.csv` without a dataframe index.

Create a new `DatasetEDA` instance if you replace `df` with a different dataframe. Notebook outputs were cleared during refactoring; run the cells in order to regenerate them.

### Before/after quality and cached reports

Notebook 00 keeps a `df_before` copy and compares it with the prepared dataframe:

```python
from src.preprocessing.cleaning import compare_dataset_quality

quality_comparison = compare_dataset_quality(df_before, df)
summary = quality_comparison["summary"]
columns = quality_comparison["columns"]
```

EDA methods, description lookup, and quality comparison automatically save JSON reports and readable CSV tables in `results/00_cleaning_and_preprocessing/`. Repeated calls reload matching results and display the same tables. Cache keys include dataframe values, index, column order, data types, report settings, helper source code, and (for descriptions) file contents and custom overrides. Data transformations such as `add_order_month` still compute normally.

Use `force_refresh=True` to recompute a report, `use_cache=False` to disable cache reads and writes, or `cache_dir=...` to change its location. `DatasetEDA` accepts these defaults in its constructor; they can also be overridden per method. Create a new EDA instance when replacing its dataframe. Deleting report files is safe: they will be generated again on the next call.

The quality summary reports structural indicators rather than a single quality score. Differences are After minus Before; percentages use each dataset's own denominator, and removing columns can itself change duplicate counts. Empty datasets have undefined percentages displayed as missing values.
