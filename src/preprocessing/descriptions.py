"""Dataset column description lookup."""

import pandas as pd
from .reporting import cached_report


@cached_report
def show_column_descriptions(
    dataframe,
    description_path,
    additional_descriptions=None,
    _report=None,
):
    """Display descriptions for selected columns, with optional custom overrides."""
    descriptions = pd.read_csv(description_path, encoding="latin-1")

    def normalize(values):
        return (
            values.astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.casefold()
        )

    descriptions["_key"] = normalize(descriptions["FIELDS"])

    selected = pd.DataFrame({"Column": dataframe.columns})
    selected["_key"] = normalize(selected["Column"])

    table = selected.merge(
        descriptions[["_key", "DESCRIPTION"]],
        on="_key",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    # Custom descriptions fill missing entries or override existing ones.
    if additional_descriptions:
        custom = pd.DataFrame(
            additional_descriptions.items(),
            columns=["Column", "Description"],
        )
        custom["_key"] = normalize(custom["Column"])
        custom_lookup = custom.set_index("_key")["Description"]

        table["DESCRIPTION"] = (
            table["_key"].map(custom_lookup)
            .combine_first(table["DESCRIPTION"])
        )

    table = table.drop(columns="_key").rename(
        columns={"DESCRIPTION": "Description"}
    )

    table["Description"] = (
        table["Description"]
        .astype("string")
        .str.replace(r"^\s*:\s*", "", regex=True)
        .str.strip()
        .fillna("Not found in the description file")
    )

    with pd.option_context("display.max_colwidth", None):
        _report.table(table)

    return table
