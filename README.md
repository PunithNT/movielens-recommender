# MovieLens Recommender: Item-Based Collaborative Filtering

A small, complete recommender system on MovieLens. It learns which movies are similar from how users rated them, recommends movies a user has not seen yet, and is evaluated honestly against a popularity baseline with precision@k, recall@k and catalog coverage.

## Problem

Streaming and rental catalogs are too large to browse. Given only past ratings, can we rank unseen movies so that the movies a user will actually like land in the top 5 or top 10? This project builds the classic first answer to that question, item-based collaborative filtering, and measures whether it beats simply recommending what everyone else watched.

## Dataset

MovieLens 100K, distributed as ml-latest-small by GroupLens.

- Download: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- Unzip so these two files exist:
  - `data/ml-latest-small/ratings.csv` with columns userId, movieId, rating, timestamp
  - `data/ml-latest-small/movies.csv` with columns movieId, title, genres
- Ratings are on a 0.5 to 5.0 scale in 0.5 steps. The real files contain about 100,000 ratings from about 600 users on about 9,000 movies.
- The downloaded data folder is gitignored. Data is never committed to this repo.

Offline fallback: if the real CSVs are not present, `src/data.py` generates deterministic MovieLens-schema synthetic data (numpy seeded, about 600 users, 900 movies with genre strings, ratings 0.5 to 5.0 with a realistic high skew, and timestamps). The synthetic data has planted genre-preference structure so the pipeline runs end to end, but it is synthetic data. Metrics computed on it are labelled synthetic in `reports/metrics.json` and must never be presented as real MovieLens results.

## Method

Item-based collaborative filtering (`src/recommender.py`, class `ItemBasedCF`):

1. Build a user by item rating matrix from the training ratings.
2. Mean-center each user's ratings by that user's mean. This removes per-user leniency, since some users rate generously and others harshly.
3. Compute item-item cosine similarity on the mean-centered matrix, treating missing entries as 0 after centering. Two movies are similar when the same users rate them similarly relative to their own averages. Similarities are then significance-weighted: a similarity estimated from only a few co-raters is shrunk toward 0, because those estimates are mostly noise.
4. Predict a rating with a weighted score: for a target item, take its k nearest neighbours (default k = 20) among items the user has rated, average the user's centered ratings on those neighbours weighted by similarity, and add the user's mean back. Predictions are clipped to 0.5 to 5.0.
5. Recommend by scoring every item the user has not rated and returning the top N. Already-rated items are always excluded.

Baseline (`PopularityRecommender`): rank items by rating count, tie-broken by mean rating, excluding items the user already rated. It has no personalization at all, which is exactly why it is a useful bar to clear.

Implementation note: the similarity is computed with a dense numpy pivot. That is sized fine for 100K ratings. scipy is listed in requirements for sparse follow-up work but is not required by the current code.

## Evaluation design

Code: `src/evaluate.py`. Run it to regenerate `reports/metrics.json`.

- Split: per-user holdout. For each user with at least 5 ratings, ratings are sorted by timestamp and the most recent 20 percent (at least 1) become that user's test set; the rest are train. Users with fewer than 5 ratings are train-only. Stated honestly: this is a per-user temporal holdout, not a random global split and not leave-one-out. It tests whether past ratings predict near-future likes, and it guarantees no test item for a user ever appears in that user's train set.
- Relevance: a held-out item is relevant if its rating is 4.0 or higher.
- Users with no relevant test items are excluded from the precision and recall averages, and the evaluated user count is reported.
- Metrics at k = 5 and k = 10:
  - Precision@k: fraction of the top k recommendations that are relevant.
  - Recall@k: fraction of the user's relevant test items found in the top k.
  - Catalog coverage at 10: fraction of all catalog movies that appear in at least one user's top 10. Low coverage means the model only ever surfaces a small slice of the catalog.
- Both item-based CF and the popularity baseline are scored on the identical split, so the comparison is apples to apples.

Reading the results: the popularity baseline is harder to beat on precision@k than it first looks, and that is a real finding rather than a bug. Held-out items are by construction items users chose to watch, and popular items are watched the most, so a holdout rewards exposure as well as taste. Where item-based CF earns its place is personalization and coverage: it produces a different list per user and surfaces far more of the catalog. In the synthetic run currently stored in `reports/metrics.json`, item-based CF and the popularity baseline are effectively tied at precision@10 (about 0.036 each), the baseline is ahead at precision@5 (about 0.033 against about 0.020), and CF coverage at 10 is about 0.95 against about 0.02 for the baseline. Those are synthetic-data numbers only, included to show the pipeline end to end. Rerun the same code on the real MovieLens files before drawing any conclusion about real-world performance.

## Quickstart

```bash
cd movielens-recommender
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: download the real data
# wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# unzip ml-latest-small.zip -d data/
# Without this step the synthetic fallback is used and labelled as such.

# Evaluate both models and write reports/metrics.json
python -m src.evaluate

# Print recommendations for one user
python scripts/recommend.py --user-id 1 --n 10
python scripts/recommend.py --user-id 1 --n 10 --model popularity

# Run the tests
pytest -q
```

## Example output

Running the CLI for a user prints a ranked list like this in structure (titles and scores depend on whether real or synthetic data is loaded, and the synthetic notice is printed first when the fallback is in use):

```text
Top 10 recommendations for user 1 [item-cf]:
 1. [movieId] Title (Year) (Genres) score=4.612
 2. [movieId] Title (Year) (Genres) score=4.540
 ...
```

Each line gives the rank, the MovieLens movieId, the title and genres from movies.csv, and the predicted rating as the score. Higher scores are stronger predicted likes, and no title in the list is one the user already rated.

## Limitations

- Cold start: a new user with no ratings gets no personalized list (the popularity baseline is the fallback), and a new movie with no ratings has no neighbours and can never be recommended.
- Sparsity: most user and movie pairs are unobserved. Similarities estimated from a handful of co-ratings are noisy, and niche movies get unreliable neighbours.
- Popularity bias: heavily rated movies have the most stable similarities, so recommendations drift toward the head of the catalog. Coverage exposes this but does not fix it.
- Ratings only: the model ignores genres, release year, timestamps beyond the split, and any context such as time of day or device.
- The temporal holdout uses one split point per user. Results vary with the split, and offline precision@k does not guarantee online click or watch behaviour.

## Next steps

- Matrix factorization with SVD on the centered matrix, compared on the same split.
- Implicit-feedback ALS using watch or rating events as confidence-weighted implicit signals.
- Hybrid scoring that blends CF with genre similarity to help cold items.
- Ranking metrics beyond precision and recall, such as NDCG@k and MAP.
- A small API or demo UI once the offline metrics justify it.

## Project structure

```text
movielens-recommender/
  README.md
  LICENSE
  requirements.txt
  app.py                  CLI entry point alias
  scripts/
    recommend.py          CLI: top-N recommendations for a user id
  src/
    data.py               Load real MovieLens CSVs or generate synthetic fallback
    recommender.py        ItemBasedCF and PopularityRecommender
    evaluate.py           Per-user split, precision@k, recall@k, coverage
  tests/
    test_recommender.py   Planted-similarity, exclusion, precision@k, split tests
  reports/
    metrics.json          Generated by python -m src.evaluate
  data/                   Real data goes in data/ml-latest-small/ (gitignored)
```

## Author

Punith Nonavinakere Thandavamurthy
M.S. Data Science, Pace University (expected Jan 2027)
GitHub: https://github.com/PunithNT
LinkedIn: linkedin.com/in/punith-nt
Email: contactpunithnt@gmail.com
