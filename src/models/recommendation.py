"""Sparse recommendation data, SVD search and final evaluation for notebook 04."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from src.config import SEED
from .latent_factors import NOTEBOOK_RESULTS, save_table

def constrained_holdout(df, random_state=42):
    """
    Hold out at most one interaction per eligible customer.

    Constraints:
    - Every customer retains at least one training interaction.
    - Every product retains at least one training interaction.
    - Each customer contributes at most one holdout interaction.
    """
    customer_counts = df.groupby('Customer Id')['Product Card Id'].transform('count')
    product_counts = df.groupby('Product Card Id')['Customer Id'].transform('count')
    eligible = df[(customer_counts >= 2) & (product_counts >= 2)].sample(frac=1, random_state=random_state)
    remaining_product_counts = df.groupby('Product Card Id').size().to_dict()
    selected_customers = set()
    holdout_indices = []
    for idx, row in eligible.iterrows():
        customer_id = row['Customer Id']
        product_id = row['Product Card Id']
        if customer_id in selected_customers:
            continue
        if remaining_product_counts[product_id] <= 1:
            continue
        holdout_indices.append(idx)
        selected_customers.add(customer_id)
        remaining_product_counts[product_id] -= 1
    holdout = df.loc[holdout_indices].copy()
    remaining = df.drop(index=holdout_indices).copy()
    return (remaining, holdout)

def evaluate_recommender(scores, R_seen, R_holdout, top_k=10):
    """
    Evaluate top-K recommendations.

    Each evaluated customer has at most one
    held-out product.
    """
    customers = np.flatnonzero(R_holdout.getnnz(axis=1) > 0)
    hits = []
    ndcgs = []
    for customer_idx in customers:
        seen_products = R_seen[customer_idx].indices
        relevant_products = R_holdout[customer_idx].indices
        if scores.ndim == 1:
            customer_scores = scores.copy()
        else:
            customer_scores = scores[customer_idx].copy()
        customer_scores[seen_products] = -np.inf
        candidates = np.flatnonzero(np.isfinite(customer_scores))
        recommended = candidates[np.argsort(customer_scores[candidates])[::-1][:top_k]]
        hit_positions = np.flatnonzero(np.isin(recommended, relevant_products))
        hit = int(len(hit_positions) > 0)
        if hit:
            rank = hit_positions[0]
            ndcg = 1 / np.log2(rank + 2)
        else:
            ndcg = 0.0
        hits.append(hit)
        ndcgs.append(ndcg)
    return {f'HitRate@{top_k}': np.mean(hits) if hits else np.nan, f'NDCG@{top_k}': np.mean(ndcgs) if ndcgs else np.nan}

class RecommendationData:
    """Aggregate interactions and keep shared customer/product sparse indices."""
    def __init__(self, frame):
        self.frame = frame.copy()
        self.interactions = (frame.groupby(["Customer Id", "Product Card Id"])
                             ["Order Item Quantity"].sum().reset_index())
        self.customer_ids = pd.Index(self.interactions["Customer Id"].unique())
        self.product_ids = pd.Index(self.interactions["Product Card Id"].unique())
        self.customer_to_idx = {value: i for i, value in enumerate(self.customer_ids)}
        self.product_to_idx = {value: i for i, value in enumerate(self.product_ids)}
        self.matrix = self.build_sparse_matrix(self.interactions)

    def build_sparse_matrix(self, frame):
        return csr_matrix((frame["Order Item Quantity"].to_numpy(dtype=np.float32),
            (frame["Customer Id"].map(self.customer_to_idx).to_numpy(),
             frame["Product Card Id"].map(self.product_to_idx).to_numpy())),
            shape=(len(self.customer_ids), len(self.product_ids)))

    def activity_summary(self):
        return pd.DataFrame({
            "Products per customer": self.interactions.groupby("Customer Id")["Product Card Id"].nunique().describe(),
            "Customers per product": self.interactions.groupby("Product Card Id")["Customer Id"].nunique().describe(),
        })

    def split(self, random_state=SEED):
        data = self.interactions.reset_index(drop=True)
        train_val, self.test = constrained_holdout(data, random_state)
        self.train, self.validation = constrained_holdout(train_val, random_state+1)
        self.R_train = self.build_sparse_matrix(self.train)
        self.R_val = self.build_sparse_matrix(self.validation)
        self.R_test = self.build_sparse_matrix(self.test)
        indices = [set(part.index) for part in (self.train, self.validation, self.test)]
        assert all(indices[i].isdisjoint(indices[j]) for i in range(3) for j in range(i+1,3))
        assert set.union(*indices) == set(data.index)
        assert (self.R_train.getnnz(axis=0) >= 1).all()
        assert (self.R_train.getnnz(axis=1) >= 1).all()
        for part in (self.validation, self.test):
            assert part.empty or part.groupby("Customer Id").size().max() <= 1
        self.split_summary = pd.DataFrame([
            {"Dataset": label, "Interactions": matrix.nnz,
             "Customers": (matrix.getnnz(axis=1)>0).sum(),
             "Products": (matrix.getnnz(axis=0)>0).sum(),
             "Percentage": 100*matrix.nnz/len(data)}
            for label, matrix in (("Train",self.R_train),("Validation",self.R_val),("Test",self.R_test))])
        save_table(self.split_summary, NOTEBOOK_RESULTS / "recommendation_split.csv")
        return self


class RecommendationSearch:
    """Evaluate SVD factor counts on validation interactions."""
    def __init__(self, k_values=(2,3,5,8,10,15,20,30,40,50), top_k=10, random_state=SEED):
        self.k_values, self.top_k, self.random_state = tuple(k_values), top_k, random_state

    def run(self, data):
        rows = []
        for k in self.k_values:
            if not 1 <= k <= min(data.R_train.shape):
                raise ValueError(f"k={k} exceeds the recommendation matrix dimensions")
            model = TruncatedSVD(n_components=k, random_state=self.random_state)
            factors = model.fit_transform(data.R_train)
            metrics = evaluate_recommender(factors @ model.components_, data.R_train,
                                           data.R_val, self.top_k)
            rows.append({"k": k, **metrics, "Explained Variance": model.explained_variance_ratio_.sum()})
            self.results = pd.DataFrame(rows)
            save_table(self.results, NOTEBOOK_RESULTS / "recommendation_svd_evaluation_by_k.csv")
        return self


class SVDRecommender:
    """Fit on train+validation before inspecting factors or evaluating test data."""
    def __init__(self, k=2, top_k=10, random_state=SEED):
        self.k, self.top_k, self.random_state = k, top_k, random_state

    def fit(self, data):
        self.data = data
        self.seen = (data.R_train + data.R_val).tocsr()
        self.model = TruncatedSVD(n_components=self.k, random_state=self.random_state)
        self.customer_factors = self.model.fit_transform(self.seen)
        return self

    def evaluate(self):
        scores = self.customer_factors @ self.model.components_
        popularity = np.asarray(self.seen.getnnz(axis=0), dtype=float).ravel()
        self.results = pd.DataFrame([
            {"Model": name, **evaluate_recommender(values,self.seen,self.data.R_test,self.top_k)}
            for name, values in (("Popularity Baseline",popularity),(f"Truncated SVD (k={self.k})",scores))])
        save_table(self.results, NOTEBOOK_RESULTS / "recommendation_test_comparison.csv")
        return self.results

    def product_factors(self):
        factors = pd.DataFrame(self.model.components_.T, index=self.data.product_ids,
                               columns=[f"Factor {i+1}" for i in range(self.k)])
        factors.index.name = "Product Card Id"
        info = self.data.frame[["Product Card Id","Product Name"]].drop_duplicates("Product Card Id").set_index("Product Card Id")
        factors = factors.join(info)
        factors["Popularity"] = self.seen.getnnz(axis=0)
        save_table(factors, NOTEBOOK_RESULTS / "recommendation_product_factors.csv", "Product Card Id")
        return factors

    def factor_extremes(self, n=10):
        factors = self.product_factors()
        rows = []
        for factor in [f"Factor {i+1}" for i in range(self.k)]:
            for direction, table in (("Highest",factors.nlargest(n,factor)),("Lowest",factors.nsmallest(n,factor))):
                table = table[["Product Name","Popularity",factor]].rename(columns={factor:"Coefficient"}).reset_index()
                table.insert(0,"Component",factor)
                table.insert(1,"Direction",direction)
                rows.append(table)
        return pd.concat(rows,ignore_index=True)

    def popularity_summary(self):
        self.popularity = self.data.interactions.groupby("Product Card Id")["Customer Id"].nunique().sort_values(ascending=False)
        self.concentration = pd.DataFrame({"Top Products":[1,5,10,20],
            "Interaction Share (%)":[100*self.popularity.head(n).sum()/self.popularity.sum() for n in (1,5,10,20)]})
        save_table(self.concentration, NOTEBOOK_RESULTS / "recommendation_popularity_concentration.csv")
        return self.concentration
