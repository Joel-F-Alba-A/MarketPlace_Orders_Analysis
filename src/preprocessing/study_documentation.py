"""Dataset dictionaries, quality comparison, and environment documentation for notebook 01."""
import platform
import sys
from importlib.metadata import distributions

import pandas as pd
from IPython.display import display, Markdown

from src.config import (RAW_DATA_PATH, CLEAN_DATA_PATH, DESCRIPTION_DATA_PATH,
                        CLEANING_RESULTS_DIR)
from .descriptions import show_column_descriptions
from .cleaning import compare_dataset_quality


class StudyDocumentation:
    """Present original and processed data without changing either source file."""

    def __init__(self, raw_path=RAW_DATA_PATH, processed_path=CLEAN_DATA_PATH,
                 description_path=DESCRIPTION_DATA_PATH, cache_dir=CLEANING_RESULTS_DIR):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.description_path = description_path
        self.cache_dir = cache_dir
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", 150)

    def load_original(self):
        """Load and preview the original CSV."""
        self.original = pd.read_csv(self.raw_path, encoding='latin-1')
        print('Original dataset shape:', self.original.shape)
        display(self.original.head())

    def describe_original(self):
        """Show all original column definitions."""
        original_feature_descriptions = show_column_descriptions(dataframe=self.original, description_path=self.description_path, additional_descriptions={'Order Zipcode': 'Order destination postal code (project clarification; omitted from the supplied dictionary).'}, cache_dir=self.cache_dir)
        return original_feature_descriptions

    def load_processed(self):
        """Load and preview the processed CSV."""
        self.processed = pd.read_csv(self.processed_path)
        print('Processed dataset shape:', self.processed.shape)
        display(self.processed.head())

    def describe_processed(self):
        """Show retained definitions and highlight analytical uses."""
        numerical_features = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Product Price']
        processed_descriptions = show_column_descriptions(dataframe=self.processed, description_path=self.description_path, additional_descriptions={'Order Month': 'Month extracted from the order date (1–12).'}, cache_dir=self.cache_dir)
        categorical_features = self.processed.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        interaction_keys = ['Customer Id', 'Product Card Id']
        used_features = set(numerical_features + categorical_features + interaction_keys)
        feature_rows = []
        for row in processed_descriptions.itertuples(index=False):
            name, description = row
            if name in numerical_features:
                use = 'Numerical model input; quantity also aggregates purchases'
            elif name in interaction_keys:
                use = 'Customer–product interaction key'
            elif name in categorical_features:
                use = 'Categorical interpretation / embedding overlays'
            else:
                use = 'Retained; not selected for these model inputs or overlays'
            label = f'**{name}**' if name in used_features else name
            description = str(description).replace('|', '/').replace('\n', ' ')
            feature_rows.append(f'| {label} | {description} | {use} |')
        display(Markdown('| Feature | Meaning | Use |\n|---|---|---|\n' + '\n'.join(feature_rows)))
        return processed_descriptions

    def compare_quality(self):
        """Show before/after structural and column quality."""
        quality_comparison = compare_dataset_quality(before=self.original, after=self.processed, cache_dir=self.cache_dir)
        return quality_comparison

    def describe_environment(self):
        """Report installed versions and package roles."""
        dependency_roles = {'numpy': 'Numerical arrays, matrix operations, and reproducible sampling.', 'pandas': 'Dataset loading, cleaning, aggregation, and result tables.', 'scipy': 'Statistics, distances, sparse matrices, and subspace comparisons.', 'scikit-learn': 'Scaling, partitions, mixtures, decomposition, and anomaly detection.', 'umap-learn': 'Nonlinear UMAP embeddings.', 'matplotlib': 'Scientific figures and comparison plots.', 'seaborn': 'Statistical plots and annotated heatmaps.', 'torch': 'AE/VAE definition, training, checkpoints, and GPU execution.', 'joblib': 'Parallel searches and model persistence.', 'threadpoolctl': 'Control numerical-library thread pools.', 'tqdm': 'Progress indicators during training and experiments.', 'mlxtend': 'Frequent itemsets and association rules (Apriori/FP-Growth).', 'ipython': 'Interactive execution and rich table/Markdown display.', 'jupyterlab': 'Notebook development and execution interface.', 'ipykernel': 'Python kernel for notebook execution.', 'nbformat': 'Read, validate, and save notebook files.', 'nbclient': 'Execute child notebooks from main.ipynb.', 'jupyter-client': 'Manage notebook kernels and their communication.'}
        print('Python:', platform.python_version())
        print('Executable:', sys.executable)
        print('Platform:', platform.platform())
        dependency_rows = []
        for distribution in distributions():
            name = distribution.metadata['Name']
            normalized_name = name.lower().replace('_', '-').replace('.', '-')
            dependency_rows.append({'Dependency': name, 'Version': distribution.version, 'Use / package description': dependency_roles.get(normalized_name, distribution.metadata.get('Summary') or 'Supporting distribution; no description supplied.')})
        dependency_versions = pd.DataFrame(dependency_rows).drop_duplicates('Dependency').sort_values('Dependency', key=lambda values: values.str.lower()).reset_index(drop=True)
        display(dependency_versions)
        return dependency_versions
