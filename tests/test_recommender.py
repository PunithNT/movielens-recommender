"""Tests for the MovieLens recommender on small synthetic data.

These tests use tiny hand-built frames, not the generated fallback
dataset, so each expected behaviour is planted and checkable.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import per_user_split, precision_at_k, recall_at_k
from src.recommender import ItemBasedCF, PopularityRecommender


def _planted_ratings() -> pd.DataFrame:
    """Movies 10 and 20 move together. Movie 30 moves against them.

    Users who love 10 also love 20 and dislike 30, and vice versa,
    so item 20 must be the most similar item to item 10.
    """
    rows = []
    # (user, rating for 10, rating for 20, rating for 30, rating for 40 filler)
    pattern = [
        (1, 5.0, 5.0, 1.0, 3.0),
        (2, 5.0, 4.5, 1.5, 3.5),
        (3, 4.5, 5.0, 1.0, 2.5),
        (4, 1.0, 1.5, 5.0, 3.0),
        (5, 1.5, 1.0, 4.5, 3.5),
        (6, 1.0, 1.0, 5.0, 2.5),
    ]
    ts = 1_600_000_000
    for user_id, r10, r20, r30, r40 in pattern:
        for movie_id, rating in [(10, r10), (20, r20), (30, r30), (40, r40)]:
            ts += 100
            rows.append(
                {"userId": user_id, "movieId": movie_id, "rating": rating, "timestamp": ts}
            )
    return pd.DataFrame(rows)


def test_similar_items_returns_planted_similar_item_first():
    model = ItemBasedCF().fit(_planted_ratings())
    similar = model.similar_items(10, n=3)
    assert similar, "expected at least one similar item"
    assert similar[0][0] == 20


def test_recommendations_exclude_already_rated_and_have_correct_length():
    ratings = _planted_ratings()
    # Add extra unrated movies so there are enough candidates.
    extra = []
    ts = 1_700_000_000
    for user_id in [2, 3, 4, 5]:
        for movie_id, rating in [(50, 4.0), (60, 2.0), (70, 4.5)]:
            ts += 100
            extra.append(
                {"userId": user_id, "movieId": movie_id, "rating": rating, "timestamp": ts}
            )
    ratings = pd.concat([ratings, pd.DataFrame(extra)], ignore_index=True)
    model = ItemBasedCF().fit(ratings)
    recs = model.recommend(1, n=3)
    assert len(recs) == 3
    rated_by_user_1 = {10, 20, 30, 40}
    assert all(movie_id not in rated_by_user_1 for movie_id, _ in recs)


def test_recommend_length_matches_request_for_popularity_baseline():
    model = PopularityRecommender().fit(_planted_ratings())
    recs = model.recommend(999, n=2)  # unknown user: nothing excluded
    assert len(recs) == 2


def test_precision_at_k_perfect_ranking_is_one():
    assert precision_at_k([1, 2, 3, 4, 5], relevant={1, 2, 3, 4, 5}, k=5) == 1.0
    assert recall_at_k([1, 2, 3, 4, 5], relevant={1, 2, 3, 4, 5}, k=5) == 1.0


def test_precision_at_k_fully_irrelevant_is_zero():
    assert precision_at_k([6, 7, 8, 9, 10], relevant={1, 2, 3}, k=5) == 0.0
    assert recall_at_k([6, 7, 8, 9, 10], relevant={1, 2, 3}, k=5) == 0.0


def test_split_does_not_leak_test_items_into_train_for_a_user():
    rows = []
    ts = 1_600_000_000
    for i in range(10):
        ts += 1000
        rows.append({"userId": 1, "movieId": 100 + i, "rating": 4.5, "timestamp": ts})
    for i in range(8):
        ts += 1000
        rows.append({"userId": 2, "movieId": 200 + i, "rating": 3.5, "timestamp": ts})
    ratings = pd.DataFrame(rows)
    train, test = per_user_split(ratings)
    for user_id in [1, 2]:
        train_items = set(train.loc[train["userId"] == user_id, "movieId"])
        test_items = set(test.loc[test["userId"] == user_id, "movieId"])
        assert train_items.isdisjoint(test_items), f"leakage for user {user_id}"
        assert test_items, f"user {user_id} should have a test set"
    # Every rating is accounted for exactly once.
    assert len(train) + len(test) == len(ratings)


def test_predict_is_within_rating_scale():
    model = ItemBasedCF().fit(_planted_ratings())
    pred = model.predict(1, 20)
    assert 0.5 <= pred <= 5.0
