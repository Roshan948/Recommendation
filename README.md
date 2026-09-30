# MovieLens‑1M Movie Recommender

## 1. Dataset & Problem
- **Dataset:** MovieLens 1M (GroupLens, Univ. of Minnesota): ~1,000,209 ratings, 6,040 users, 3,883 movies (3,706 rated), collected 2000–2003.
- **Files:** `ratings.dat` (user, movie, 1–5 stars, timestamp), `users.dat` (gender, age bucket, occupation, zip), `movies.dat` (title + year, genres).
- **Problem:** predict the rating a user would give a movie, and use those predictions to recommend unseen movies.
- **Challenges:** 95.5 % of the user×movie matrix is empty, ratings skew high (mean ≈ 3.58), popularity is a long tail, and new users have no history (cold start).

## 2. What I Did
- Loaded and inspected all three files, then ran EDA (rating distribution, activity per user/movie, genres, demographics, time span).
- Cleaned the data: missing values, duplicates, invalid ranges, orphan IDs, inconsistent titles and zips, outliers.
- Engineered features and vectorized text and categorical fields.
- Scaled numeric features, then trained and compared baselines, linear, tree‑based and neural models.
- Saved the best model and preprocessing objects, and built a Gradio GUI (plus CLI) that runs from the saved files.

## 3. Why These Decisions
- **Random 80/10/10 split:** standard for ML‑1M; validation is used for tuning and early stopping, the test set is scored only once.
- **Heavy users and rare movies kept:** they are genuine, not errors. Their influence is reduced with log counts and shrunk (Bayesian) means.
- **Leave‑one‑out means on training rows:** the user/movie mean features exclude the row's own rating, so the label cannot leak into the features. Val/test rows use train‑only statistics.
- **Title fixes:** the year is extracted as a number, "Matrix, The" becomes "The Matrix", and alternate‑language names are dropped so TF‑IDF sees clean words.
- **Zip → region:** zip formats were inconsistent, so only the first digit is kept (10 = unknown).
- **Scaling only where needed:** StandardScaler and one‑hot for the linear and neural inputs, fitted on the training set only.
- **Neural hybrid:** matrix factorization captures personal taste, and an MLP adds side information (demographics, genres, title words, year), which also helps sparse users.

## 4. How It Was Implemented
- **Features:** shrunk user mean and movie mean, log rating counts, gender, age, occupation and region, release year, 18 genre flags, 20 SVD components of title TF‑IDF.
- **Vectorization:** `MultiLabelBinarizer` for genres, `TfidfVectorizer` + `TruncatedSVD` for titles, `OneHotEncoder` for occupation/region.
- **Models compared:**
  - Baselines: global mean; user mean + movie mean.
  - Ridge regression (3 alphas).
  - HistGradientBoosting (2 configs).
  - PyTorch Matrix Factorization (2 configs).
  - PyTorch Hybrid NN: embeddings + biases + MLP (3 configs).
- **Training:** AdamW, mini‑batches of 4,096, early stopping on validation RMSE (patience 4), best weights restored.
- **Metrics:** RMSE (main), MAE, R².
- **Saved to `artifacts/`:** `model.pt`, `model_config.json`, `preprocessing.joblib`, `users_lookup.csv`, `movies_lookup.csv`, `ratings_clean.csv.gz`, `experiment_results.csv`.
- **Prototype (`app.py`):**
  - Top‑N recommendations for an existing user, with a genre filter.
  - Rating prediction for a user + movie.
  - Similar movies (learned embeddings or genre/title content).
  - New‑user cold start: rate up to 5 movies and get recommendations without retraining.

## 5. Key Results & Findings
- **Data:** ratings skew positive (4 is the most common; ~57 % are 4–5 stars). Film‑Noir, Documentary and War rate highest, Horror lowest. ~90 % of ratings were made in 2000.
- **Cleaning:** ML‑1M was already clean (no missing values or duplicates); the main fixes were titles, zips and outlier handling.
- **Baselines (validation RMSE):** global mean ≈ 1.12; user + movie mean ≈ 0.93. Most of the gain comes from knowing who is rating and what is rated.
- **Feature models:** Ridge ≈ 0.92 and HistGradientBoosting ≈ 0.92, only a small step above the additive baseline.
- **Neural models:** the collaborative models beat the feature‑only models, and the Hybrid NN was best. **Add your exact numbers from `artifacts/experiment_results.csv`:**

  | Model | Val RMSE | Test RMSE | Test MAE | Test R² |
  |---|---|---|---|---|
  | Best Hybrid NN | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

- **Takeaway:** personalised latent factors matter most; side information adds a smaller further gain and helps sparse users.
- **Limitations:**
  - Random split, not a time‑based one.
  - The model is trained on the train split only.
  - Brand‑new users need the fold‑in method.
  - Movies with fewer than 20 training ratings are never recommended.

## 6. Run It
```bash
pip install numpy pandas scipy scikit-learn joblib torch gradio
python app.py --cli --user 1 --n 10   # terminal test
python app.py                         # GUI at http://127.0.0.1:7860
```
Keep `app.py` next to the `artifacts/` folder, or set `ART_DIR` to its path.
