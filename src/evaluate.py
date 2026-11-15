"""Evaluation: per-user holdout with precision@k, recall@k, coverage.

Split design (stated honestly):
- For each user with at least 5 ratings, ratings are sorted by
  timestamp and the most recent 20 percent (at least 1 rating) are
  held out as that user's test set. The rest are train.
- Users with fewer than 5 ratings contribute only to train. Their
  taste is too thin to evaluate fairly, and holding out would leave
  almost nothing to learn from.
- This is a per-user temporal holdout, not a random global split and
  not a leave-one-out split. It answers: given a user's past, can we
  rank their near-future likes highly?

Relevance: a test item is relevant if its held-out rating is >= 4.0.
Users with no relevant test items are excluded from the precision
and recall averages (there is nothing to find for them), and that
count is reported in the metrics file.

Catalog coverage: fraction of all catalog items that appear in at
least one user's top-k recommendation list.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.recommender import ItemBasedCF, PopularityRecommender

RELEVANCE_THRESHOLD = 4.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0
    hits = sum(1 for item in topk if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for item in topk if item in relevant)
    return hits / len(relevant)


def per_user_split(
    ratings: pd.DataFrame, test_fraction: float = 0.2, min_ratings: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal holdout. Returns (train, test)."""
    train_parts, test_parts = [], []
    for _, group in ratings.sort_values("timestamp").groupby("userId"):
        if len(group) < min_ratings:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(len(group) * test_fraction)))
        n_test = min(n_test, len(group) - 1)
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    train = pd.concat(train_parts).reset_index(drop=True) if train_parts else ratings.iloc[0:0]
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else ratings.iloc[0:0]
    return train, test


def _evaluate_model(model, train: pd.DataFrame, test: pd.DataFrame, catalog_size: int, ks=(5, 10)) -> dict:
    model.fit(train)
    max_k = max(ks)
    relevant_by_user: dict[int, set] = {}
    for user_id, group in test.groupby("userId"):
        rel = set(group.loc[group["rating"] >= RELEVANCE_THRESHOLD, "movieId"].astype(int))
        if rel:
            relevant_by_user[int(user_id)] = rel

    metrics: dict[str, float] = {}
    covered: set[int] = set()
    per_k_hits = {k: {"p": [], "r": []} for k in ks}
    for user_id, relevant in relevant_by_user.items():
        recs = [movie_id for movie_id, _ in model.recommend(user_id, n=max_k)]
        covered.update(recs[:max_k])
        for k in ks:
            per_k_hits[k]["p"].append(precision_at_k(recs, relevant, k))
            per_k_hits[k]["r"].append(recall_at_k(recs, relevant, k))

    for k in ks:
        metrics[f"precision@{k}"] = float(np.mean(per_k_hits[k]["p"])) if per_k_hits[k]["p"] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(per_k_hits[k]["r"])) if per_k_hits[k]["r"] else 0.0
    metrics["coverage@10"] = len(covered) / catalog_size if catalog_size else 0.0
    metrics["users_evaluated"] = len(relevant_by_user)
    return metrics


def run_evaluation(data_dir=None) -> dict:
    ratings, movies, is_synthetic = load_data(data_dir)
    train, test = per_user_split(ratings)
    catalog_size = int(movies["movieId"].nunique())

    cf_metrics = _evaluate_model(ItemBasedCF(k_neighbors=20), train, test, catalog_size)
    pop_metrics = _evaluate_model(PopularityRecommender(), train, test, catalog_size)

    report = {
        "data_source": "synthetic_movielens_schema" if is_synthetic else "movielens_ml_latest_small",
        "is_synthetic": bool(is_synthetic),
        "note": (
            "Metrics on synthetic data illustrate the pipeline only and are "
            "not real MovieLens results."
            if is_synthetic
            else "Metrics computed on the real MovieLens ml-latest-small data."
        ),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "split": "per_user_temporal_holdout_20pct_min_5_ratings",
        "n_ratings_total": int(len(ratings)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_users": int(ratings["userId"].nunique()),
        "n_movies_catalog": catalog_size,
        "item_based_cf": cf_metrics,
        "popularity_baseline": pop_metrics,
    }
    return report


def main() -> None:
    report = run_evaluation()
    out = PROJECT_ROOT / "reports" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
