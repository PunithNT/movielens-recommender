"""Item-based collaborative filtering and a popularity baseline.

Method (item-based CF):
- Build a user by item rating matrix.
- Mean-center each user's ratings by that user's mean rating. This
  removes per-user leniency (some users rate everything high).
- Compute item-item cosine similarity on the centered matrix,
  treating missing entries as 0 after centering, then apply
  significance weighting: a similarity is shrunk toward 0 when the
  pair was co-rated by few users, since those estimates are noisy.
- For prediction, take the k nearest neighbours of the target item
  among items the user has rated and form a similarity-weighted
  average of the user's centered ratings, then add the user mean back.
- For recommendation, score every unrated item this way (in practice
  via its neighbours) and return the highest scoring items.

The implementation uses a dense pivot with numpy and pandas. That is
sized fine for MovieLens 100K (about 600 users by 9000 movies at most,
and about 600 by 900 in the synthetic fallback). scipy is not required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ItemBasedCF:
    def __init__(self, k_neighbors: int = 20, shrinkage: float = 25.0):
        self.k_neighbors = k_neighbors
        # Similarities from few co-raters are noisy. Significance
        # weighting shrinks a similarity toward 0 when the pair was
        # co-rated by fewer than about `shrinkage` users.
        self.shrinkage = shrinkage
        self.user_ids_: np.ndarray | None = None
        self.movie_ids_: np.ndarray | None = None
        self.user_index_: dict = {}
        self.movie_index_: dict = {}
        self.ratings_matrix_: np.ndarray | None = None  # raw, NaN = missing
        self.centered_: np.ndarray | None = None
        self.user_means_: np.ndarray | None = None
        self.similarity_: np.ndarray | None = None
        self.similarity_knn_: np.ndarray | None = None
        self.global_mean_: float = 3.5

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        pivot = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self.user_ids_ = pivot.index.to_numpy()
        self.movie_ids_ = pivot.columns.to_numpy()
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.movie_index_ = {m: i for i, m in enumerate(self.movie_ids_)}
        self.ratings_matrix_ = pivot.to_numpy(dtype=float)
        self.global_mean_ = float(np.nanmean(self.ratings_matrix_))
        self.user_means_ = np.nanmean(self.ratings_matrix_, axis=1)
        # Users with no ratings cannot occur in a pivot, but guard anyway.
        self.user_means_ = np.where(
            np.isnan(self.user_means_), self.global_mean_, self.user_means_
        )
        centered = self.ratings_matrix_ - self.user_means_[:, None]
        self.centered_ = np.where(np.isnan(centered), 0.0, centered)

        # Item-item cosine similarity on the centered matrix.
        norms = np.linalg.norm(self.centered_, axis=0)
        norms[norms == 0] = 1.0
        normalized = self.centered_ / norms[None, :]
        self.similarity_ = normalized.T @ normalized
        if self.shrinkage:
            rated_mask = (~np.isnan(self.ratings_matrix_)).astype(float)
            co_counts = rated_mask.T @ rated_mask
            self.similarity_ = self.similarity_ * (
                co_counts / (co_counts + self.shrinkage)
            )
        np.fill_diagonal(self.similarity_, 1.0)

        # Neighbour-masked copy: per item, keep only the k strongest
        # neighbours by absolute similarity. Scoring with every item
        # instead of the nearest neighbours drowns the signal in noise
        # from barely co-rated pairs, so recommend() and predict()
        # both use this masked matrix for the weighted average.
        self.similarity_knn_ = np.zeros_like(self.similarity_)
        if self.k_neighbors and self.similarity_.shape[0] > 1:
            for i in range(self.similarity_.shape[0]):
                row = np.abs(self.similarity_[i])
                ranked = np.argsort(-row)
                keep = [j for j in ranked if j != i][: self.k_neighbors]
                self.similarity_knn_[i, keep] = self.similarity_[i, keep]
        else:
            self.similarity_knn_ = self.similarity_.copy()
        return self

    def _check_fitted(self) -> None:
        if self.similarity_ is None:
            raise RuntimeError("Call fit() before using the recommender.")

    def similar_items(self, movie_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the n most similar movies to movie_id as (id, score)."""
        self._check_fitted()
        if movie_id not in self.movie_index_:
            return []
        idx = self.movie_index_[movie_id]
        sims = self.similarity_[idx].copy()
        sims[idx] = -np.inf  # exclude itself
        order = np.argsort(-sims)[:n]
        return [(int(self.movie_ids_[j]), float(sims[j])) for j in order]

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a rating for (user_id, movie_id) on the 0.5 to 5.0 scale."""
        self._check_fitted()
        if movie_id not in self.movie_index_:
            return float(self.global_mean_)
        if user_id not in self.user_index_:
            # Cold user: fall back to the item mean, then the global mean.
            col = self.ratings_matrix_[:, self.movie_index_[movie_id]]
            item_mean = np.nanmean(col)
            return float(self.global_mean_ if np.isnan(item_mean) else item_mean)

        u = self.user_index_[user_id]
        m = self.movie_index_[movie_id]
        rated_mask = ~np.isnan(self.ratings_matrix_[u])
        if not rated_mask.any():
            return float(self.user_means_[u])

        # similarity_knn_ already keeps only the k nearest neighbours.
        sims = self.similarity_knn_[m].copy()
        sims[~rated_mask] = 0.0

        denom = np.abs(sims).sum()
        if denom == 0:
            return float(self.user_means_[u])
        pred_centered = float(sims @ self.centered_[u] / denom)
        return float(np.clip(self.user_means_[u] + pred_centered, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return top n (movie_id, predicted_score) for a user.

        Items the user has already rated are always excluded. Unknown
        users get an empty list from this class; callers who need a
        cold-start answer should use PopularityRecommender instead.
        """
        self._check_fitted()
        if user_id not in self.user_index_:
            return []
        u = self.user_index_[user_id]
        rated_mask = ~np.isnan(self.ratings_matrix_[u])

        # Score all items at once with the neighbour-weighted formula,
        # using only each item's k nearest neighbours.
        # sim @ centered gives, per item, sum_j sim(item, j) * centered_j.
        weighted = self.similarity_knn_ @ self.centered_[u]
        denom = np.abs(self.similarity_knn_) @ rated_mask.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(denom > 0, weighted / np.where(denom == 0, 1, denom), 0.0)
        scores = self.user_means_[u] + scores
        scores[rated_mask] = -np.inf
        if np.all(np.isneginf(scores)):
            return []
        order = np.argsort(-scores)
        results = []
        for j in order:
            if np.isneginf(scores[j]):
                break
            results.append((int(self.movie_ids_[j]), float(np.clip(scores[j], 0.5, 5.0))))
            if len(results) >= n:
                break
        return results


class PopularityRecommender:
    """Baseline: recommend the most rated items, tie-broken by mean rating.

    Items the target user has already rated are excluded when a
    ratings frame with that user is known from fit().
    """

    def __init__(self):
        self.ranking_: list[int] = []
        self.scores_: dict[int, float] = {}
        self.user_rated_: dict[int, set] = {}

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(["count", "mean"])
            .reset_index()
            .sort_values(["count", "mean"], ascending=[False, False])
        )
        self.ranking_ = stats["movieId"].astype(int).tolist()
        self.scores_ = dict(
            zip(stats["movieId"].astype(int), stats["mean"].astype(float))
        )
        self.user_rated_ = (
            ratings.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int))).to_dict()
        )
        return self

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        seen = self.user_rated_.get(user_id, set())
        out = []
        for movie_id in self.ranking_:
            if movie_id in seen:
                continue
            out.append((movie_id, self.scores_.get(movie_id, 0.0)))
            if len(out) >= n:
                break
        return out
