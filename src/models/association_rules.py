"""Customer purchase histories and Apriori association rules for notebook 08."""
import pandas as pd
from IPython.display import display
from mlxtend.frequent_patterns import apriori, association_rules
from src.config import CLEAN_DATA_PATH


class AssociationData:
    """Keep original records separate from aggregated and binary matrices."""

    def __init__(self, path=CLEAN_DATA_PATH, frame=None):
        self.frame = pd.read_csv(path) if frame is None else frame.copy(deep=True)

    def quantity_preview(self, rows=10, columns=10):
        """Aggregate total purchased quantity by customer and product."""
        self.quantities = pd.crosstab(
            index=self.frame["Customer Id"], columns=self.frame["Product Card Id"],
            values=self.frame["Order Item Quantity"], aggfunc="sum",
        ).fillna(0).astype(int)
        return self.quantities.iloc[:rows, :columns]

    def binary_preview(self, rows=10, columns=10):
        """Build purchase-presence input for customer-history association rules."""
        self.matrix = self.quantities > 0
        return self.matrix.iloc[:rows, :columns]

    def describe(self):
        print("Customers:", self.matrix.shape[0])
        print("Products:", self.matrix.shape[1])

    def describe_support(self):
        self.product_support = self.matrix.mean(axis=0)
        print(self.product_support.describe())
        print("\nProducts with support >= 1%:")
        print((self.product_support >= 0.01).sum())
        print("\nProducts with support >= 5%:")
        print((self.product_support >= 0.05).sum())

    def products_in_rules(self, rules, count=10):
        """Return original IDs, names, and categories in both sides of top rules."""
        ids = set()
        for column in ["antecedents", "consequents"]:
            for itemset in rules.head(count)[column]:
                ids.update(itemset)
        return (self.frame.loc[self.frame["Product Card Id"].isin(ids),
                               ["Product Card Id", "Product Name", "Category Name"]]
                .drop_duplicates().sort_values(["Category Name", "Product Card Id"])
                .reset_index(drop=True))


class AprioriRules:
    """Mine the existing notebook settings and present ID/name-based rules."""

    def __init__(self, data, min_support=0.05, min_confidence=0.50):
        self.data = data
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit_itemsets(self):
        self.frequent_itemsets = apriori(
            self.data.matrix, min_support=self.min_support,
            use_colnames=True, low_memory=True,
        )
        print("Frequent itemsets:", len(self.frequent_itemsets))
        print("\nItemset sizes:")
        print(self.frequent_itemsets["itemsets"].apply(len).value_counts().sort_index())
        return self

    def itemsets_of_size(self, size=9):
        self.frequent_itemsets["size"] = self.frequent_itemsets["itemsets"].apply(len)
        return self.frequent_itemsets[self.frequent_itemsets["size"] == size]

    def generate_rules(self):
        self.rules = association_rules(
            self.frequent_itemsets, metric="confidence",
            min_threshold=self.min_confidence, num_itemsets=len(self.data.matrix),
        )
        print("Number of rules:", len(self.rules))
        return self

    def summarize(self, count=10):
        self.rules_summary = self.rules[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ].copy().sort_values(by="lift", ascending=False)
        display(self.rules_summary.head(count))
        return self.rules_summary

    def positive(self, count=10):
        self.positive_rules = self.rules_summary[self.rules_summary["lift"] > 1].copy()
        print("Positive association rules:", len(self.positive_rules))
        display(self.positive_rules.head(count))
        return self.positive_rules

    def readable(self, use_names=False, count=10):
        """Format a copy; preserve original itemsets for later product lookup."""
        names = (self.data.frame.drop_duplicates("Product Card Id")
                 .set_index("Product Card Id")["Product Name"].to_dict()) if use_names else {}

        def format_itemset(itemset):
            return ", ".join(str(names.get(product_id, product_id))
                             for product_id in sorted(itemset))

        result = self.positive_rules.copy()
        for column in ["antecedents", "consequents"]:
            result[column] = result[column].apply(format_itemset)
        if use_names:
            pd.set_option("display.max_colwidth", None)
        display(result.head(count))
        return result
