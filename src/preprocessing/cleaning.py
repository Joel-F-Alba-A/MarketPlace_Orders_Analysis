"""Reusable data cleaning and feature preparation functions."""

import pandas as pd


def add_order_month(
    dataframe,
    date_column="order date (DateOrders)",
    output_column="Order Month",
):
    """Return a copy with the order month extracted from the date column."""
    result = dataframe.copy()
    result[output_column] = pd.to_datetime(result[date_column]).dt.month
    return result


from .reporting import cached_report


def _quality_metrics(dataframe):
    """Compute structural quality indicators, not a subjective quality score."""
    import numpy as np

    rows, columns = dataframe.shape
    missing = dataframe.isna()
    unique_counts = dataframe.nunique(dropna=True)
    duplicate_count = int(dataframe.duplicated().sum())
    text = dataframe.select_dtypes(include=["object", "string", "category"])
    blank_count = sum(
        int(text[column].astype("string").str.strip().eq("").fillna(False).sum())
        for column in text.columns
    )
    numeric = dataframe.select_dtypes(include="number")
    infinite_count = int(numeric.isin([np.inf, -np.inf]).sum().sum())
    return pd.Series({
        "Rows": rows,
        "Columns": columns,
        "Missing cells": int(missing.sum().sum()),
        "Missing cells (%)": missing.sum().sum() / dataframe.size * 100 if dataframe.size else float("nan"),
        "Rows with missing values": int(missing.any(axis=1).sum()),
        "Duplicate rows after first": duplicate_count,
        "Duplicate rows (%)": duplicate_count / rows * 100 if rows else float("nan"),
        "Constant columns (one non-null value)": int(unique_counts.eq(1).sum()),
        "All-null columns": int(unique_counts.eq(0).sum()),
        "Blank or whitespace-only text cells": blank_count,
        "Infinite numeric values": infinite_count,
        "Memory (MiB)": dataframe.memory_usage(deep=True).sum() / 1024**2,
    })


@cached_report
def compare_dataset_quality(before, after, _report=None):
    """Display and return before/after quality indicators and column changes.

    Return a dictionary with summary and columns tables. Differences mean
    After - Before; removing columns can change percentages and duplicates.
    Reports are cached by both dataframes' contents. Optional keyword arguments:
    cache_dir, use_cache, force_refresh. Neither input dataframe is modified.
    """
    summary = pd.concat(
        [_quality_metrics(before).rename("Before"),
         _quality_metrics(after).rename("After")], axis=1
    )
    summary["Difference (After - Before)"] = summary["After"] - summary["Before"]
    summary.index.name = "Metric"

    def column_metrics(dataframe, suffix):
        table = pd.DataFrame({
            f"Type {suffix}": dataframe.dtypes.astype(str),
            f"Missing Count {suffix}": dataframe.isna().sum(),
            f"Missing (%) {suffix}": dataframe.isna().mean().mul(100),
            f"Unique Count {suffix}": dataframe.nunique(dropna=True),
        })
        return table

    columns = column_metrics(before, "Before").join(
        column_metrics(after, "After"), how="outer"
    )
    # Preserve the original column order, with new columns appended.
    order = before.columns.append(after.columns[~after.columns.isin(before.columns)])
    columns = columns.reindex(order)
    columns.insert(0, "Status", [
        "Retained" if name in before.columns and name in after.columns
        else "Removed" if name in before.columns else "Added"
        for name in columns.index
    ])
    columns.index.name = "Column"
    _report.message("Dataset quality: before and after")
    _report.table(summary)
    _report.message("Column changes and quality indicators")
    _report.table(columns)
    return {"summary": summary, "columns": columns}
