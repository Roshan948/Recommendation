import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.preprocessing import normalize

ART_DIR = os.environ.get("ART_DIR", "artifacts")
MIN_TRAIN_COUNT = 20      

class TorchScorer:
    """Rebuilds the network described in model_config.json and exposes numpy-friendly scoring."""

    def __init__(self, art_dir):
        import torch
        import torch.nn as nn

        cfg = json.load(open(os.path.join(art_dir, "model_config.json")))

        class MatrixFactorization(nn.Module):
            def __init__(self, n_users, n_movies, emb_dim):
                super().__init__()
                self.user_emb = nn.Embedding(n_users, emb_dim)
                self.movie_emb = nn.Embedding(n_movies, emb_dim)
                self.user_bias = nn.Embedding(n_users, 1)
                self.movie_bias = nn.Embedding(n_movies, 1)
                self.register_buffer("global_mean", torch.tensor(0.0))

            def forward(self, u, m):
                dot = (self.user_emb(u) * self.movie_emb(m)).sum(dim=1)
                return self.global_mean + self.user_bias(u).squeeze(-1) + self.movie_bias(m).squeeze(-1) + dot

        class HybridRecommender(nn.Module):
            def __init__(self, n_users, n_movies, n_user_num, n_movie_num, emb_dim=32, hidden=128,
                         dropout=0.2, n_occ=21, n_reg=11):
                super().__init__()
                self.register_buffer("user_cat", torch.zeros(n_users, 2, dtype=torch.long))
                self.register_buffer("user_num", torch.zeros(n_users, n_user_num))
                self.register_buffer("movie_num", torch.zeros(n_movies, n_movie_num))
                self.register_buffer("global_mean", torch.tensor(0.0))
                self.user_emb = nn.Embedding(n_users, emb_dim)
                self.movie_emb = nn.Embedding(n_movies, emb_dim)
                self.user_bias = nn.Embedding(n_users, 1)
                self.movie_bias = nn.Embedding(n_movies, 1)
                self.occ_emb = nn.Embedding(n_occ, 8)
                self.reg_emb = nn.Embedding(n_reg, 4)
                width = 2 * emb_dim + 8 + 4 + n_user_num + n_movie_num
                self.mlp = nn.Sequential(
                    nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, u, m):
                ue, me = self.user_emb(u), self.movie_emb(m)
                occ_id, reg_id = self.user_cat[u].unbind(dim=1)
                side = torch.cat([ue, me, self.occ_emb(occ_id), self.reg_emb(reg_id),
                                  self.user_num[u], self.movie_num[m]], dim=1)
                collab = (ue * me).sum(dim=1) + self.user_bias(u).squeeze(-1) + self.movie_bias(m).squeeze(-1)
                return self.global_mean + collab + self.mlp(side).squeeze(-1)

        if cfg["architecture"] == "HybridRecommender":
            model = HybridRecommender(cfg["n_users"], cfg["n_movies"], cfg["n_user_num"], cfg["n_movie_num"],
                                      cfg["emb_dim"], cfg["hidden"], cfg["dropout"])
        else:
            model = MatrixFactorization(cfg["n_users"], cfg["n_movies"], cfg["emb_dim"])
        model.load_state_dict(torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu"))
        model.eval()

        self.torch, self.model, self.cfg = torch, model, cfg
        self.gmean = float(cfg["global_mean"])
        self.n_movies = cfg["n_movies"]
        self._all_movies = torch.arange(self.n_movies)
        self.Q = model.movie_emb.weight.detach().numpy().astype(np.float64)          # movie vectors
        self.bi = model.movie_bias.weight.detach().numpy().ravel().astype(np.float64)  # movie biases

    def score_user(self, u_idx):
        """Predicted rating (clipped to 1-5) of every movie for one existing user."""
        with self.torch.no_grad():
            users = self.torch.full((self.n_movies,), int(u_idx), dtype=self.torch.long)
            return self.model(users, self._all_movies).clamp(1, 5).numpy().astype(np.float64)


# ----------------------------------------------------------------------------------------------
# Recommendation logic (pure numpy / pandas - independent of the network implementation)
# ----------------------------------------------------------------------------------------------
class Recommender:
    def __init__(self, scorer, art_dir=ART_DIR):
        self.scorer = scorer
        self.gmean = float(scorer.gmean)
        prep = joblib.load(os.path.join(art_dir, "preprocessing.joblib"))
        self.uid2idx, self.mid2idx = prep["uid2idx"], prep["mid2idx"]

        self.users = pd.read_csv(os.path.join(art_dir, "users_lookup.csv")).sort_values("u_idx").reset_index(drop=True)
        self.movies = pd.read_csv(os.path.join(art_dir, "movies_lookup.csv")).sort_values("m_idx").reset_index(drop=True)
        n_users, n_movies = len(self.users), len(self.movies)

        # movie labels shown in the dropdowns (made unique if two movies share title + year)
        labels = self.movies.Title + " (" + self.movies.Year.astype(str) + ")"
        dup = labels.duplicated(keep=False)
        labels[dup] = labels[dup] + " [id " + self.movies.MovieID.astype(str)[dup] + "]"
        self.movies["Label"] = labels
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.labels = labels.tolist()

        # everything each user has ever rated (used to hide already-seen movies and to show history)
        r = pd.read_csv(os.path.join(art_dir, "ratings_clean.csv.gz"))
        self.R = sp.csr_matrix((r.Rating.to_numpy(float), (r.UserID.map(self.uid2idx).to_numpy(),
                                                            r.MovieID.map(self.mid2idx).to_numpy())),
                               shape=(n_users, n_movies))

        # content vectors for "similar movies": saved genre binarizer + saved title TF-IDF
        genre_lists = [g.split("|") for g in self.movies.Genres]
        self.G = prep["genre_binarizer"].transform(genre_lists).astype(float)
        self.genre_names = list(prep["genre_binarizer"].classes_)
        title_vec = prep["title_tfidf"].transform(self.movies.Title)
        self.content = normalize(sp.hstack([sp.csr_matrix(self.G), 0.5 * title_vec]).tocsr())
        self.Qn = normalize(scorer.Q)

        cnt = self.movies.train_count.to_numpy(float)
        self.eligible = cnt >= MIN_TRAIN_COUNT
        avg = self.movies.avg_rating.fillna(0).to_numpy(float)
        self.pop_score = (avg * cnt + PRIOR_W * self.gmean) / (cnt + PRIOR_W)   # shrunk mean, cold-start fallback

    # ---------------------------------------------------------------- helpers
    def _genre_mask(self, genres):
        if not genres:
            return np.ones(len(self.movies), bool)
        cols = [self.genre_names.index(g) for g in genres]
        return self.G[:, cols].sum(1) > 0

    def _table(self, idx, scores, score_label):
        idx = np.asarray(idx, dtype=int)
        out = pd.DataFrame({
            "#": np.arange(1, len(idx) + 1),
            "Movie": self.movies.Label.to_numpy()[idx],
            "Genres": self.movies.Genres.to_numpy()[idx],
            "Avg rating": self.movies.avg_rating.round(2).to_numpy()[idx],
            score_label: np.round(scores, 2),
        })
        return out

    @staticmethod
    def _top(scores, n):
        order = np.argsort(-scores)[: int(n)]
        order = order[np.isfinite(scores[order])]
        return order, scores[order]

    def _user_index(self, user_id):
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            raise ValueError("Please enter a numeric user id.")
        if uid not in self.uid2idx:
            raise ValueError(f"Unknown user id {uid}. Valid ids run from 1 to {len(self.users)}.")
        return self.uid2idx[uid]

    # ---------------------------------------------------------------- features
    def profile(self, user_id):
        u = self._user_index(user_id)
        row = self.users.iloc[u]
        return (f"User {int(row.UserID)}: {row.Gender}, age {row.AgeGroup}, {row.OccupationName}, "
                f"{int(row.n_ratings)} ratings")

    def recommend(self, user_id, n=10, genres=None):
        u = self._user_index(user_id)
        scores = self.scorer.score_user(u).copy()
        seen = self.R[u].indices
        scores[seen] = -np.inf
        scores[~self.eligible] = -np.inf
        scores[~self._genre_mask(genres)] = -np.inf
        idx, sc = self._top(scores, n)
        recs = self._table(idx, sc, "Predicted rating")

        row = self.R[u]
        order = np.lexsort((-self.movies.train_count.to_numpy()[row.indices], -row.data))[:10]
        history = self._table(row.indices[order], row.data[order], "Your rating")
        return recs, history

    def predict(self, user_id, movie_label):
        u = self._user_index(user_id)
        if movie_label not in self.label2idx:
            raise ValueError("Please pick a movie from the list.")
        m = self.label2idx[movie_label]
        pred = float(self.scorer.score_user(u)[m])
        actual = self.R[u, m]
        return pred, (float(actual) if actual else None)

    def similar(self, movie_label, n=10, mode="collaborative"):
        if movie_label not in self.label2idx:
            raise ValueError("Please pick a movie from the list.")
        m = self.label2idx[movie_label]
        if mode == "collaborative":
            sim = self.Qn @ self.Qn[m]
        else:
            sim = (self.content @ self.content[m].T).toarray().ravel()
        sim = sim.astype(float).copy()
        sim[m] = -np.inf
        sim[~self.eligible] = -np.inf
        idx, sc = self._top(sim, n)
        return self._table(idx, sc, "Similarity")

    def new_user(self, genres, rated, n=10):
        """rated: list of (movie_label, stars). Folds the user into the learned movie-embedding space."""
        pairs = [(self.label2idx[l], float(r)) for l, r in rated if l in self.label2idx]
        Q, bi = self.scorer.Q, self.scorer.bi
        if pairs:
            idx = np.array([p[0] for p in pairs])
            stars = np.array([p[1] for p in pairs])
            resid = stars - self.gmean - bi[idx]
            bias = resid.sum() / (len(resid) + PRIOR_W)
            Qr = Q[idx]
            p = np.linalg.solve(Qr.T @ Qr + FOLD_LAMBDA * np.eye(Q.shape[1]), Qr.T @ (resid - bias))
            scores = np.clip(self.gmean + bias + bi + Q @ p, 1, 5)
        else:
            idx = np.array([], dtype=int)
            scores = self.pop_score.copy()
        scores[idx] = -np.inf                       # do not re-recommend the movies just rated
        scores[~self.eligible] = -np.inf
        scores[~self._genre_mask(genres)] = -np.inf
        top, sc = self._top(scores, n)
        return self._table(top, sc, "Score")


# ----------------------------------------------------------------------------------------------
# Gradio GUI
# ----------------------------------------------------------------------------------------------
def build_ui(rec):
    import gradio as gr

    genres = rec.genre_names
    labels = rec.labels
    empty = pd.DataFrame({"Info": ["-"]})

    def do_recommend(user_id, n, picked):
        try:
            recs, hist = rec.recommend(user_id, n, picked)
            return rec.profile(user_id), recs, hist
        except ValueError as e:
            return str(e), empty, empty

    def do_predict(user_id, label):
        try:
            pred, actual = rec.predict(user_id, label)
        except ValueError as e:
            return str(e)
        stars = "★" * int(round(pred)) + "☆" * (5 - int(round(pred)))
        text = f"**{label}**\n\nPredicted rating for this user: **{pred:.2f} / 5**  {stars}"
        if actual is not None:
            text += f"\n\n_The user has already rated this movie: {actual:.0f} stars._"
        return text

    def do_similar(label, n, mode):
        try:
            key = "collaborative" if mode.startswith("Collab") else "content"
            return rec.similar(label, n, key)
        except ValueError as e:
            return pd.DataFrame({"Info": [str(e)]})

    def do_new_user(picked, *args):
        rated = [(t, r) for t, r in zip(args[0::2], args[1::2]) if t]
        return rec.new_user(picked, rated, 10)

    with gr.Blocks(title="MovieLens Recommender") as demo:
        gr.Markdown("# MovieLens-1M Recommender\nHybrid neural recommender (embeddings + side information), "
                    "served from the artifacts saved by the notebook.")
        with gr.Tab("Recommend for a user"):
            uid = gr.Number(value=1, precision=0, label=f"User ID (1-{len(rec.users)})")
            n = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="How many recommendations")
            gsel = gr.CheckboxGroup(genres, label="Only these genres (optional)")
            b1 = gr.Button("Recommend", variant="primary")
            prof = gr.Markdown()
            o1 = gr.Dataframe(label="Recommended (unseen) movies")
            o2 = gr.Dataframe(label="This user's highest-rated movies")
            b1.click(do_recommend, [uid, n, gsel], [prof, o1, o2])
        with gr.Tab("Predict a rating"):
            u2 = gr.Number(value=1, precision=0, label="User ID")
            m2 = gr.Dropdown(labels, label="Movie")
            b2 = gr.Button("Predict", variant="primary")
            o3 = gr.Markdown()
            b2.click(do_predict, [u2, m2], o3)
        with gr.Tab("Similar movies"):
            m3 = gr.Dropdown(labels, label="Movie")
            n3 = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="How many")
            md = gr.Radio(["Collaborative (learned embeddings)", "Content (genres + title words)"],
                          value="Collaborative (learned embeddings)", label="Similarity type")
            b3 = gr.Button("Find similar", variant="primary")
            o4 = gr.Dataframe()
            b3.click(do_similar, [m3, n3, md], o4)
        with gr.Tab("New user (cold start)"):
            gr.Markdown("Rate up to five movies you know; the app places you in the learned taste space "
                        "without retraining.")
            g4 = gr.CheckboxGroup(genres, label="Only these genres (optional)")
            inputs = [g4]
            for i in range(5):
                with gr.Row():
                    d = gr.Dropdown(labels, label=f"Movie {i + 1}", value=None)
                    s = gr.Slider(minimum=1, maximum=5, value=4, step=1, label="Your rating")
                    inputs += [d, s]
            b4 = gr.Button("Recommend for me", variant="primary")
            o5 = gr.Dataframe()
            b4.click(do_new_user, inputs, o5)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true", help="print recommendations in the terminal instead of the GUI")
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--n", type=int, default=10)
    args, _ = ap.parse_known_args()

    rec = Recommender(TorchScorer(ART_DIR), ART_DIR)
    if args.cli:
        recs, hist = rec.recommend(args.user, args.n)
        print(rec.profile(args.user))
        print("\nTop rated by this user:\n", hist.head(5).to_string(index=False))
        print("\nRecommendations:\n", recs.to_string(index=False))
        return
    demo = build_ui(rec)
    demo.launch(share="google.colab" in sys.modules)


if __name__ == "__main__":
    main()
