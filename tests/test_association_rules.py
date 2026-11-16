"""Regression checks for customer-history aggregation and association semantics."""
import contextlib
import io
import unittest

import pandas as pd
from src.models.association_rules import AssociationData, AprioriRules


class AssociationRulesTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'Customer Id': [1, 1, 1, 2, 2, 3, 4],
            'Product Card Id': [10, 10, 20, 10, 20, 10, 30],
            'Order Item Quantity': [2, 3, 1, 1, 1, 1, 1],
            'Product Name': ['A', 'A', 'B', 'A', 'B', 'A', 'C'],
            'Category Name': ['X', 'X', 'Y', 'X', 'Y', 'X', 'Z'],
        })
        self.data = AssociationData(frame=self.frame)
        self.data.quantity_preview()
        self.data.binary_preview()

    def test_repeat_purchases_count_once_for_support(self):
        self.assertEqual(self.data.quantities.loc[1, 10], 5)
        self.assertEqual(self.data.matrix.shape, (4, 3))
        self.assertAlmostEqual(self.data.matrix[10].mean(), .75)
        pd.testing.assert_frame_equal(self.data.frame, self.frame)

    def test_rule_metrics_and_name_formatting_preserve_itemsets(self):
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = AprioriRules(self.data, min_support=.25, min_confidence=.5)
            analysis.fit_itemsets().generate_rules()
            analysis.summarize()
            analysis.positive()
            readable = analysis.readable(use_names=True)
        row = analysis.rules[
            analysis.rules.antecedents.eq(frozenset({10}))
            & analysis.rules.consequents.eq(frozenset({20}))
        ].iloc[0]
        self.assertAlmostEqual(row.support, .5)
        self.assertAlmostEqual(row.confidence, 2/3)
        self.assertAlmostEqual(row.lift, 4/3)
        self.assertTrue(all(isinstance(x, frozenset) for x in analysis.rules.antecedents))
        self.assertTrue(all(isinstance(x, str) for x in readable.antecedents))
        products = self.data.products_in_rules(analysis.positive_rules)
        self.assertEqual(set(products['Product Card Id']), {10, 20})


if __name__ == '__main__':
    unittest.main()
