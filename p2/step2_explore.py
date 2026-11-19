"""
Step 2: what do the embeddings capture that matters for emoji prediction, and what do they miss?

  A. Nearest neighbours of task words, GloVe Twitter 200d vs fastText news 300d (+ overlap of the two lists)
  B. Emoji vectors: their nearest neighbours show whether the vector is meaningful
  C. Class centroids: average tweet vector per emoji, cosine similarity between classes (heatmap),
     and a nearest-centroid classifier (no training beyond averaging) as a "pure similarity" score
  D. Limitations: antonyms are neighbours, one vector per word, colours, negation lost by averaging
  E. How often those limitations actually occur in our tweets (which ones are troubling for the task)

Writes results/step2_explore.json and results/figures/step2_*.png.
Run:  .venv/bin/python -m p2.step2_explore
"""
import json
import re

import matplotlib.pyplot as plt
import numpy as np

from p2.common import EMOJIS, NAMES, RESULTS, load_data, load_embeddings, metrics

MODELS = ["glove200", "fasttext300"]                    # compared in detail (A, B, D)
CENTROID_MODELS = MODELS + ["glovewiki200", "word2vec300"]   # nearest-centroid score for all four (C)
TASK_WORDS = ["love", "happy", "lol", "christmas", "fire", "lit", "beach", "sun",
              "photo", "america", "kiss", "babe", "blessed", "purple", "blue", "100"]
PAIRS = [("love", "hate"), ("happy", "sad"), ("good", "bad"), ("blue", "purple"),
         ("photo", "picture"), ("camera", "photo"), ("lol", "haha"), ("christmas", "xmas")]


def mean_vector(text, emb):
    vecs = [emb[w] for w in emb.tokenize(text) if w in emb]
    return np.mean(vecs, axis=0) if vecs else np.zeros(emb.dim, dtype=np.float32)


def unit(m):
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)


def main():
    data = load_data()
    embs = {m: load_embeddings(m) for m in CENTROID_MODELS}
    out = {}

    # A. nearest neighbours
    out["neighbours"] = {}
    for w in TASK_WORDS:
        row = {m: [x for x, _ in embs[m].neighbours(w, k=8)] if w in embs[m] else None for m in MODELS}
        both = [set(embs[m].neighbours(w, k=10)) if w in embs[m] else set() for m in MODELS]
        row["overlap@10"] = len({x for x, _ in both[0]} & {x for x, _ in both[1]})
        out["neighbours"][w] = row

    # B. emoji vectors: having a vector is not the same as having a *good* vector
    out["emoji_neighbours"] = {
        m: {e: [w for w, _ in embs[m].neighbours(e, k=6)] for e in ["❤", "✨", "☀", "😂", "🔥"] if e in embs[m]}
        for m in MODELS}

    # C. class centroids on the training tweets
    out["centroids"] = {}
    y_tr, y_va = data["train"][1], data["validation"][1]
    for m in CENTROID_MODELS:
        X_tr = np.stack([mean_vector(t, embs[m]) for t in data["train"][0]])
        X_va = np.stack([mean_vector(t, embs[m]) for t in data["validation"][0]])
        mu = X_tr.mean(0)                                   # remove the shared "tweet" direction
        C = unit(np.stack([X_tr[y_tr == c].mean(0) for c in range(20)]) - mu)
        sim = C @ C.T
        pred = (unit(X_va - mu) @ C.T).argmax(1)
        off = sim - np.eye(20) * 2
        top = sorted({tuple(sorted(p)) for p in zip(*np.unravel_index(np.argsort(-off, axis=None)[:16], off.shape))},
                     key=lambda p: -sim[p])[:6]
        out["centroids"][m] = {"nearest_centroid_val": metrics(y_va, pred),
                               "most_similar_classes": [(EMOJIS[a] + EMOJIS[b], round(float(sim[a, b]), 2)) for a, b in top]}
        if m == "glove200":
            fig, ax = plt.subplots(figsize=(6.5, 6))
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(20), NAMES, fontsize=7, rotation=90)
            ax.set_yticks(range(20), NAMES, fontsize=7)
            ax.set_title("Cosine similarity of class centroids (GloVe Twitter 200d)", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
            (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(RESULTS / "figures" / "step2_centroid_similarity.png", dpi=200)
            plt.close(fig)
            out["centroid_matrix_glove200"] = np.round(sim, 3).tolist()

    # D. limitations
    out["pair_similarity"] = {f"{a}-{b}": {m: round(embs[m].similarity(a, b), 2) for m in MODELS} for a, b in PAIRS}
    out["negation"] = {}
    for a, b in [("i am so happy today", "i am not happy today"), ("this is good", "this is not good")]:
        out["negation"][f"{a} | {b}"] = {
            m: round(float(unit(mean_vector(a, embs[m])) @ unit(mean_vector(b, embs[m]))), 3) for m in MODELS}

    # E. how often do the limitations actually hit our tweets? (training tweets)
    texts, y = data["train"][0], data["train"][1]
    has = lambda pattern: np.array([bool(re.search(pattern, t.lower())) for t in texts])   # noqa: E731
    negation, hundred = has(r"\b(not|no|never)\b|n't"), has(r"\b100\b")
    out["how_often"] = {
        "tweets_with_negation": round(float(negation.mean()), 3),
        "hundred_tweets_containing_100": round(float(hundred[y == 15].mean()), 3),
        "blue_heart_tweets_with_word_blue": round(float(has(r"\bblue\b")[y == 8].mean()), 3),
        "purple_heart_tweets_with_word_purple": round(float(has(r"\bpurple\b")[y == 13].mean()), 3),
    }

    (RESULTS / "step2_explore.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # print a readable summary
    print("A. nearest neighbours (GloVe Twitter | fastText news)")
    for w, r in out["neighbours"].items():
        print(f"  {w:<10} G: {', '.join(r['glove200'] or ['-'])}\n  {'':<10} F: {', '.join(r['fasttext300'] or ['-'])}   overlap@10={r['overlap@10']}")
    print("\nB. nearest neighbours of emoji vectors")
    for m, d in out["emoji_neighbours"].items():
        for e, ws in d.items():
            print(f"  {m:<12} {e}  {' '.join(ws)}")
    print("\nC. class centroids")
    for m, r in out["centroids"].items():
        print(f"  {m}: nearest-centroid validation {r['nearest_centroid_val']}  most similar: {r['most_similar_classes']}")
    print("\nD. pair similarities", json.dumps(out["pair_similarity"]))
    print("   negation", json.dumps(out["negation"]))
    print("\nE. how often", json.dumps(out["how_often"]))


if __name__ == "__main__":
    main()
