"""CLI: print top-N recommendations for a user id.

Usage:
    python scripts/recommend.py --user-id 1 --n 10
    python scripts/recommend.py --user-id 42 --n 5 --model popularity

If real MovieLens files are not present under data/ml-latest-small/,
deterministic MovieLens-schema synthetic data is used and that fact
is printed at the top of the output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.recommender import ItemBasedCF, PopularityRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens recommendations for a user")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", choices=["item-cf", "popularity"], default="item-cf")
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    ratings, movies, is_synthetic = load_data(args.data_dir)
    if is_synthetic:
        print("Note: using MovieLens-schema synthetic data (real files not found).")
    titles = dict(zip(movies["movieId"].astype(int), movies["title"]))
    genres = dict(zip(movies["movieId"].astype(int), movies["genres"]))

    model = ItemBasedCF() if args.model == "item-cf" else PopularityRecommender()
    model.fit(ratings)
    recs = model.recommend(args.user_id, n=args.n)
    if not recs:
        print(f"No recommendations for user {args.user_id} (unknown user or no candidates).")
        print("Tip: for a cold user, try --model popularity.")
        return
    print(f"Top {len(recs)} recommendations for user {args.user_id} [{args.model}]:")
    for rank, (movie_id, score) in enumerate(recs, start=1):
        print(f"{rank:2d}. [{movie_id}] {titles.get(movie_id, 'Unknown title')} "
              f"({genres.get(movie_id, '')}) score={score:.3f}")


if __name__ == "__main__":
    main()
