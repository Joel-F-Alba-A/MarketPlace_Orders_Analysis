"""Reusable exploratory checks that display results and return summary tables."""

from itertools import combinations

import numpy as np
import pandas as pd
from .reporting import DEFAULT_RESULTS_DIR, cached_report


class DatasetEDA:
    """Analyze a dataframe without modifying it.

    Create a new instance after replacing the dataframe in the notebook.
    """

    def __init__(self, dataframe, cache_dir=None, use_cache=True, force_refresh=False):
        self.dataframe = dataframe
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_RESULTS_DIR
        self.use_cache = use_cache
        self.force_refresh = force_refresh

    @cached_report
    def dimensions(self, _report=None):
        """Display dataset dimensions and per-column memory usage."""
        df = self.dataframe
        _report.message(f"Rows: {df.shape[0]:,}")
        _report.message(f"Columns: {df.shape[1]:,}")
        _report.message(f"Total cells: {df.size:,}")
        _report.message(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MiB")

        _report.table(
            df.memory_usage(index=True, deep=True)
            .div(1024**2)
            .rename("Memory (MiB)")
            .sort_values(ascending=False)
            .to_frame()
        )

    @cached_report
    def missing_values(self, _report=None):
        """Display missing counts and percentages."""
        df = self.dataframe
        missing_values = pd.DataFrame({
            "Missing Count": df.isna().sum(),
            "Missing Percentage": df.isna().mean().mul(100).round(2)
        }).sort_values("Missing Count", ascending=False)

        _report.table(missing_values)

        _report.message(f"Total missing values: {df.isna().sum().sum():,}")
        _report.message(f"Rows with missing values: {df.isna().any(axis=1).sum():,}")
        return missing_values

    @cached_report
    def duplicate_records(self, _report=None):
        """Display duplicate counts and all members of duplicate groups."""
        df = self.dataframe
        # Count exact duplicate rows after their first occurrence.
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = df.duplicated().mean() * 100

        _report.message(f"Duplicate rows: {duplicate_count:,}")
        _report.message(f"Duplicate percentage: {duplicate_percentage:.2f}%")

        # Display all rows belonging to duplicate groups.
        _report.table(df.loc[df.duplicated(keep=False)])
        return df.loc[df.duplicated(keep=False)]

    @cached_report
    def identical_columns(self, _report=None):
        """Find columns equal row by row, including matching missing values."""
        df = self.dataframe
        # Find columns with identical values in the same rows.
        # Missing values in the same positions count as equal.
        identical_columns = []

        for column_a, column_b in combinations(df.columns, 2):
            a = df[column_a]
            b = df[column_b]

            equal_values = a.eq(b).fillna(False)
            both_missing = a.isna() & b.isna()

            if (equal_values | both_missing).all():
                identical_columns.append({
                    "Column A": column_a,
                    "Column B": column_b,
                    "Identical Rows": len(df),
                    "Type A": str(a.dtype),
                    "Type B": str(b.dtype)
                })

        identical_columns = pd.DataFrame(
            identical_columns,
            columns=["Column A", "Column B", "Identical Rows", "Type A", "Type B"]
        )

        _report.message(f"Pairs of identical columns: {len(identical_columns)}")
        _report.table(identical_columns)
        return identical_columns

    @cached_report
    def unique_values(self, _report=None):
        """Display unique counts, constant columns, and all-null columns."""
        df = self.dataframe
        unique_values = pd.DataFrame({
            "Unique Count": df.nunique(dropna=True),
            "Missing Count": df.isna().sum()
        }).sort_values("Unique Count")

        _report.table(unique_values)

        # Columns with exactly one distinct non-null value.
        constant_columns = unique_values.index[unique_values["Unique Count"].eq(1)]

        _report.message(f"Constant columns: {len(constant_columns)}")
        if len(constant_columns):
            _report.table(df.loc[:, constant_columns].describe(include="all").T)
        else:
            _report.table(pd.DataFrame())

        # Columns containing only missing values.
        empty_columns = unique_values.index[unique_values["Unique Count"].eq(0)]

        _report.message(f"All-null columns: {len(empty_columns)}")
        _report.message(empty_columns.tolist())
        return unique_values

    @cached_report
    def numerical_statistics(self, _report=None):
        """Display numerical descriptive statistics."""
        df = self.dataframe
        numerical_df = df.select_dtypes(include="number")

        numerical_stats = numerical_df.describe(
            percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
        ).T

        numerical_stats["median"] = numerical_df.median()
        numerical_stats["skewness"] = numerical_df.skew()
        numerical_stats["kurtosis"] = numerical_df.kurt()

        _report.table(numerical_stats)
        return numerical_stats

    @cached_report
    def multiplicative_relationships(self, relative_tolerance=1e-5, absolute_tolerance=0.01, min_valid_rows=30, _report=None):
        """Find A * B = C relationships on all valid rows using the given tolerances."""
        df = self.dataframe
        numeric_data = (
            df.select_dtypes(include="number")
            .replace([np.inf, -np.inf], np.nan)
        )

        # Exclude empty and constant columns.
        numeric_data = numeric_data.loc[
            :, numeric_data.nunique(dropna=True).gt(1)
        ]

        # Adjust these for the precision of your dataset.
        RELATIVE_TOLERANCE = relative_tolerance
        ABSOLUTE_TOLERANCE = absolute_tolerance
        MIN_VALID_ROWS = min_valid_rows

        results = []

        for column_a, column_b in combinations(numeric_data.columns, 2):
            with np.errstate(over="ignore", invalid="ignore"):
                product = numeric_data[column_a] * numeric_data[column_b]

            for target in numeric_data.columns:
                if target in (column_a, column_b):
                    continue

                # Use rows where both inputs and the target are available.
                valid = (
                    np.isfinite(product)
                    & numeric_data[column_a].notna()
                    & numeric_data[column_b].notna()
                    & numeric_data[target].notna()
                )

                valid_count = int(valid.sum())

                if valid_count < MIN_VALID_ROWS:
                    continue

                actual = numeric_data.loc[valid, target].to_numpy(dtype=float)
                predicted = product.loc[valid].to_numpy(dtype=float)

                # Skip comparisons that are constant on the available rows.
                if np.unique(actual).size < 2 or np.unique(predicted).size < 2:
                    continue

                matches = np.isclose(
                    predicted,
                    actual,
                    rtol=RELATIVE_TOLERANCE,
                    atol=ABSOLUTE_TOLERANCE,
                )

                # Report relationships that hold on every valid row.
                if matches.all():
                    results.append({
                        "Column A": column_a,
                        "Column B": column_b,
                        "Target Column": target,
                        "Relationship": f"[{column_a}] * [{column_b}] = [{target}]",
                        "Rows Checked": valid_count,
                        "Coverage (%)": round(valid_count / len(df) * 100, 2),
                        "Maximum Absolute Error": np.abs(predicted - actual).max(),
                    })

        multiplicative_dependencies = pd.DataFrame(
            results,
            columns=[
                "Column A",
                "Column B",
                "Target Column",
                "Relationship",
                "Rows Checked",
                "Coverage (%)",
                "Maximum Absolute Error",
            ],
        )

        _report.message(f"Multiplicative relationships found: {len(multiplicative_dependencies)}")

        with pd.option_context("display.max_colwidth", None):
            _report.table(multiplicative_dependencies)
        return multiplicative_dependencies
