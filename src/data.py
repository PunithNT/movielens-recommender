"""Data loading for the MovieLens recommender.

Loads real MovieLens CSVs when they are present under
data/ml-latest-small/. When they are not present, generates
deterministic synthetic data in the exact MovieLens schema so the
whole project runs offline.

The synthetic fallback is labelled everywhere it is used. It is
MovieLens-schema synthetic data, not real MovieLens data, and any
metrics computed on it are synthetic-data metrics only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def _synthetic_movies(n_movies: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for movie_id in range(1, n_movies + 1):
        n_genres = int(rng.integers(1, 4))
        genres = "|".join(rng.choice(GENRES, size=n_genres, replace=False))
        year = int(rng.integers(1960, 2025))
        rows.append(
            {
                "movieId": movie_id,
                "title": f"Synthetic Movie {movie_id} ({year})",
                "genres": genres,
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_data(
    n_users: int = 600,
    n_movies: int = 900,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic ratings in the MovieLens schema.

    Returns (ratings, movies) where:
    - ratings has columns userId, movieId, rating, timestamp
    - movies has columns movieId, title, genres
    - ratings are on the 0.5 to 5.0 scale in 0.5 steps, skewed high
    - each movie belongs to a latent genre cluster and each user has
      latent genre preferences, so collaborative filtering has real
      structure to learn from

    This is MovieLens-schema synthetic data, not real MovieLens data.
    """
    rng = np.random.default_rng(seed)
    movies = _synthetic_movies(n_movies=n_movies, seed=seed)

    # Latent structure: assign each movie a primary cluster from its
    # first genre, and give each user a preference per genre.
    primary_genre = movies["genres"].str.split("|").str[0].tolist()
    genre_index = {g: i for i, g in enumerate(GENRES)}
    movie_cluster = np.array([genre_index[g] for g in primary_genre])

    # Movie quality: some movies are simply better liked than others.
    movie_quality = rng.normal(0.0, 0.45, size=n_movies)

    # Popularity: a moderately skewed lognormal draw. A few movies get
    # many ratings, but the head is not so extreme that the popularity
    # baseline wins by construction.
    popularity = np.exp(rng.normal(0.0, 0.35, size=n_movies))

    # User preferences over genres, plus a per-user leniency offset.
    user_prefs = rng.normal(0.0, 0.75, size=(n_users, len(GENRES)))
    user_offset = rng.normal(0.15, 0.25, size=n_users)

    base_ts = 1_450_000_000  # around Dec 2015, plausible MovieLens range
    records = []
    for user_idx in range(n_users):
        user_id = user_idx + 1
        n_ratings = int(rng.integers(80, 160))
        # Users mostly watch movies in genres they like, plus popular
        # titles. Selection weights blend both, which is how real
        # rating data is generated: exposure is not random.
        taste_weight = np.exp(1.4 * user_prefs[user_idx, movie_cluster])
        weights = popularity * taste_weight
        weights = weights / weights.sum()
        chosen = rng.choice(
            n_movies, size=min(n_ratings, n_movies), replace=False, p=weights
        )
        for movie_idx in chosen:
            latent = (
                3.20
                + user_offset[user_idx]
                + movie_quality[movie_idx]
                + user_prefs[user_idx, movie_cluster[movie_idx]]
                + rng.normal(0.0, 0.45)
            )
            # Round to the nearest 0.5 and clip to the MovieLens scale.
            rating = float(np.clip(np.round(latent * 2) / 2, 0.5, 5.0))
            timestamp = int(base_ts + rng.integers(0, 200_000_000))
            records.append(
                {
                    "userId": user_id,
                    "movieId": int(movie_idx + 1),
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )

    ratings = pd.DataFrame(records).sort_values(
        ["userId", "timestamp"]
    ).reset_index(drop=True)
    return ratings, movies


def load_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Load ratings and movies.

    Returns (ratings, movies, is_synthetic).
    - If data_dir (or the default data/ml-latest-small/) contains
      ratings.csv and movies.csv, those real files are loaded and
      is_synthetic is False.
    - Otherwise deterministic synthetic data is generated and
      is_synthetic is True.
    """
    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    ratings_path = folder / "ratings.csv"
    movies_path = folder / "movies.csv"
    if ratings_path.exists() and movies_path.exists():
        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)
        ratings = ratings[["userId", "movieId", "rating", "timestamp"]]
        movies = movies[["movieId", "title", "genres"]]
        return ratings, movies, False

    ratings, movies = generate_synthetic_data()
    return ratings, movies, True


if __name__ == "__main__":
    ratings_df, movies_df, synthetic = load_data()
    source = "synthetic (MovieLens schema)" if synthetic else "real MovieLens"
    print(f"Source: {source}")
    print(f"Ratings: {len(ratings_df)} rows, {ratings_df['userId'].nunique()} users")
    print(f"Movies: {len(movies_df)} rows")
    print(ratings_df.head())
