"""
Step 6: error analysis. Which characteristics of a tweet predict where the three designs differ?

Systems compared on the same 50k test tweets (each is the ensemble of its 15 CV models, chosen before testing):
    P1 BoW      : Project 1's best, logistic regression on bag-of-words + bigrams (majority vote)
    P2 pool     : step 3, frozen GloVe, mean+max pooling, MLP (averaged probabilities)
    P2 RNN      : step 5's best recurrent model (averaged probabilities)

Analysis features (one row per test tweet), each with the reason we expected it to matter:
    n_tokens        length: RNNs can use word order, but long inputs are harder to compress into one vector
    unseen_rate     share of words never seen in the 45k training tweets: BoW knows nothing about them,
                    pretrained vectors do (step 3's hypothesis)
    oov_rate        share of tokens without a GloVe vector (these are zero vectors for P2 models)
    log_freq        mean log training frequency of the words: rare vocabulary
    location        the tweet ends with an Instagram-style "@ Place" (49% of tweets): photo/place emojis
    n_hashtags      hashtags are glued words (#sundayfunday), often unknown to GloVe
    negation        not / no / never / n't: averaging ignores it, an RNN can in principle use it
    sentiment       (#positive - #negative opinion words) / n_tokens, Hu & Liu lexicon (as in the tutorial)
    intensity       elongation, repeated !!!, ALL CAPS (GloVe's <elong> <repeat> <allcaps> tags)
    user_mention    @user present: tweets addressed to someone (kiss / wink emojis)

Analyses:
  1. per-bin macro-F1 / accuracy for each feature and system           -> results/step6_bins.json, figure
  2. the course's Bayesian correctness model (part2/analysis.py): logistic regression of "ensemble prediction
     correct?" on the standardised features, one weight vector per system  -> results/step6_bda.json, figures
  3. per-emoji F1 for each system, confusion matrix of the RNN, confident mistakes, and the tweets where
     the RNN and the pooling model disagree                            -> results/step6_examples.json, figures

Run:  .venv/bin/python -m p2.step6_errors
"""
import json
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from p2.common import EMOJIS, NAMES, RESULTS, glove_tokenize, load_data, load_embeddings, metrics
from p2.step4_rnn import setting_name
from p2.step5_search import cv_candidates

FIG = RESULTS / "figures"
NEGATIONS = {"not", "no", "never", "n't", "nothing", "nobody", "none", "cannot"}
FEATURES = ["n_tokens", "unseen_rate", "oov_rate", "log_freq", "location", "n_hashtags", "negation",
            "sentiment", "intensity", "user_mention"]

# ------------------------------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------------------------------

def system_predictions(y_test):
    p1 = np.stack([np.load(RESULTS / "p1_test_preds" / f"bow+bigrams__logistic__s{s}__f{f}.npy")
                   for s in range(3) for f in range(5)])
    preds = {"P1 BoW": np.array([np.bincount(c, minlength=20).argmax() for c in p1.T]),
             "P2 pool": np.load(RESULTS / "step3_cv" / "test_ensemble_probs.npy").astype(np.float32).argmax(1),
             "P2 RNN": np.load(RESULTS / "step5_cv" / setting_name(cv_candidates()["best"])
                               / "test_ensemble_probs.npy").astype(np.float32).argmax(1)}
    for name, p in preds.items():
        print(f"{name:<8} test {metrics(y_test, p)}")
    return preds

# ------------------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------------------

def tweet_features(texts, train_texts, emb):
    from nltk.corpus import opinion_lexicon
    pos, neg = set(opinion_lexicon.positive()), set(opinion_lexicon.negative())
    train_freq = Counter(w for t in train_texts for w in glove_tokenize(t))
    rows = []
    for t in texts:
        toks = glove_tokenize(t)
        words = [w for w in toks if w.isalpha()] or toks
        n = max(len(toks), 1)
        rows.append({
            "n_tokens": len(toks),
            "unseen_rate": sum(train_freq[w] == 0 for w in toks) / n,
            "oov_rate": sum(w not in emb for w in toks) / n,
            "log_freq": float(np.mean([np.log1p(train_freq[w]) for w in words])),
            "location": int(bool(re.search(r"\s@\s+\S", t))),
            "n_hashtags": t.count("#"),
            "negation": int(any(w in NEGATIONS for w in toks)),
            "sentiment": (sum(w in pos for w in toks) - sum(w in neg for w in toks)) / n,
            "intensity": int(any(w in ("<elong>", "<repeat>", "<allcaps>") for w in toks)),
            "user_mention": int("<user>" in toks),
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------------------
# 1. Binned scores
# ------------------------------------------------------------------------------------------

def bins_for(feature, x):
    if x.nunique() <= 2:
        return x.map({0: "no", 1: "yes"})
    if feature == "n_tokens":
        return pd.cut(x, [0, 8, 12, 16, 22, 100], labels=["≤8", "9-12", "13-16", "17-22", ">22"])
    if feature in ("unseen_rate", "oov_rate"):
        return pd.cut(x, [-1, 0, 0.05, 0.1, 1], labels=["0", "≤5%", "5-10%", ">10%"])
    if feature == "n_hashtags":
        return pd.cut(x, [-1, 0, 1, 3, 100], labels=["0", "1", "2-3", "4+"])
    if feature == "sentiment":
        return pd.cut(x, [-2, -1e-9, 1e-9, 2], labels=["negative", "neutral", "positive"])
    return pd.qcut(x, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")


def binned_scores(df, preds, y):
    out = []
    for feat in FEATURES:
        b = bins_for(feat, df[feat])
        for level in b.cat.categories if hasattr(b, "cat") else sorted(b.unique()):
            m = (b == level).to_numpy()
            row = {"feature": feat, "bin": str(level), "n": int(m.sum())}
            for name, p in preds.items():
                s = metrics(y[m], p[m])
                row[f"{name} macro_f1"], row[f"{name} acc"] = s["macro_f1"], s["accuracy"]
            out.append(row)
    return pd.DataFrame(out)


def plot_bins(table, preds):
    feats = ["n_tokens", "unseen_rate", "location", "negation", "n_hashtags", "sentiment"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    colors = {"P1 BoW": "#9aa5b1", "P2 pool": "#f2a541", "P2 RNN": "#1f4e79"}
    for ax, feat in zip(axes.flat, feats):
        t = table[table.feature == feat]
        x = np.arange(len(t))
        for i, name in enumerate(preds):
            ax.bar(x + (i - 1) * 0.27, t[f"{name} macro_f1"] * 100, 0.27, label=name, color=colors[name])
        ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(t.bin, t.n)], fontsize=7)
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("test macro-F1 (%)", fontsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "step6_bins.png", dpi=200)

# ------------------------------------------------------------------------------------------
# 2. Bayesian correctness model (course part2/analysis.py)
# ------------------------------------------------------------------------------------------

def run_bda(df, preds, y, n_items=10_000, seed=0):
    from numpyro.diagnostics import summary as summary_stats
    from part2.analysis import fit_bda_model

    items = np.random.default_rng(seed).choice(len(y), n_items, replace=False)   # same tweets for every system
    systems = list(preds)
    X = df[FEATURES].to_numpy(dtype=float)[items]
    sys_id = np.repeat(np.arange(len(systems)), n_items)
    correct = np.concatenate([(preds[s][items] == y[items]).astype(int) for s in systems])
    mcmc = fit_bda_model(len(systems), sys_id, np.tile(X, (len(systems), 1)), correct, num_chains=2, seed=seed)
    post = mcmc.get_samples()
    n_div = int(mcmc.get_extra_fields()["diverging"].sum())
    diag = summary_stats(mcmc.get_samples(group_by_chain=True))
    rhat = float(max(np.max(diag[v]["r_hat"]) for v in ("w", "sys_shift", "global_mean")))

    w = np.asarray(post["w"])                                    # (samples, systems, features)
    summary = {"n_items": n_items, "divergences": n_div, "max_rhat": rhat,
               "accuracy": {s: {"mean": float(np.asarray(post["accuracy"])[:, i].mean()),
                                "hdi95": np.percentile(np.asarray(post["accuracy"])[:, i], [2.5, 97.5]).tolist()}
                            for i, s in enumerate(systems)},
               "weights": {f: {s: {"mean": float(w[:, i, j].mean()),
                                   "hdi95": np.percentile(w[:, i, j], [2.5, 97.5]).tolist(),
                                   "P(>0)": float((w[:, i, j] > 0).mean())}
                               for i, s in enumerate(systems)} for j, f in enumerate(FEATURES)},
               # where do the designs DIFFER? posterior of the weight difference between systems
               "weight_differences": {f: {f"{b} - {a}": {"mean": float((w[:, ib, j] - w[:, ia, j]).mean()),
                                                         "P(>0)": float((w[:, ib, j] - w[:, ia, j] > 0).mean())}
                                          for ia, a in enumerate(systems) for ib, b in enumerate(systems) if ib > ia}
                                      for j, f in enumerate(FEATURES)}}

    # forest plot: features on the y axis, one interval per system
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#9aa5b1", "#f2a541", "#1f4e79"]
    for j, f in enumerate(FEATURES):
        for i, s in enumerate(systems):
            lo, hi = np.percentile(w[:, i, j], [2.5, 97.5])
            yy = j + (i - 1) * 0.22
            ax.plot([lo, hi], [yy, yy], color=colors[i], lw=2, label=s if j == 0 else None)
            ax.plot(np.median(w[:, i, j]), yy, "o", color=colors[i], ms=4)
    ax.axvline(0, color="grey", lw=0.7)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.invert_yaxis()
    ax.set_xlabel("weight on log-odds of a correct prediction (standardised feature), 95% interval")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Bayesian correctness model: which tweets does each system get right?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "step6_bda_weights.png", dpi=200)
    return summary

# ------------------------------------------------------------------------------------------
# 3. Per-emoji scores, confusions, examples
# ------------------------------------------------------------------------------------------

def per_class_and_examples(texts, df, preds, y, probs_rnn):
    per_class = pd.DataFrame({s: f1_score(y, p, average=None) for s, p in preds.items()}, index=NAMES)
    per_class["RNN - pool"] = per_class["P2 RNN"] - per_class["P2 pool"]

    cm = confusion_matrix(y, preds["P2 RNN"], normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmax=0.5)
    ax.set_xticks(range(20), NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(20), NAMES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("P2 RNN: confusion matrix (row-normalised, test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "step6_confusion_rnn.png", dpi=200)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:8], off.shape))[0]

    conf = probs_rnn.max(1)
    wrong = np.where(preds["P2 RNN"] != y)[0]
    confident_wrong = wrong[np.argsort(-conf[wrong])][:15]
    rnn_only = np.where((preds["P2 RNN"] == y) & (preds["P2 pool"] != y))[0]
    pool_only = np.where((preds["P2 RNN"] != y) & (preds["P2 pool"] == y))[0]

    def show(ids):
        return [{"text": texts[i], "true": EMOJIS[y[i]], "rnn": EMOJIS[preds["P2 RNN"][i]],
                 "pool": EMOJIS[preds["P2 pool"][i]], "bow": EMOJIS[preds["P1 BoW"][i]],
                 "rnn_conf": round(float(conf[i]), 3)} for i in ids]

    rng = np.random.default_rng(0)
    disagreement = {"rnn_right_pool_wrong": int(len(rnn_only)), "pool_right_rnn_wrong": int(len(pool_only)),
                    # are the tweets where only the RNN is right different from those where only pooling is?
                    "feature_means": {
                        "rnn_only": df.iloc[rnn_only][FEATURES].mean().round(3).to_dict(),
                        "pool_only": df.iloc[pool_only][FEATURES].mean().round(3).to_dict(),
                        "all": df[FEATURES].mean().round(3).to_dict()}}
    return {"per_class_f1": per_class.round(3).to_dict(),
            "top_confusions_rnn": [{"true": NAMES[a], "pred": NAMES[b], "share": round(float(cm[a, b]), 3)}
                                   for a, b in top],
            "confident_mistakes_rnn": show(confident_wrong),
            "disagreement": disagreement,
            "examples_rnn_right_pool_wrong": show(rng.choice(rnn_only, 10, replace=False)),
            "examples_pool_right_rnn_wrong": show(rng.choice(pool_only, 10, replace=False))}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = load_data()
    texts, y = data["test"]
    emb = load_embeddings("glove200")
    preds = system_predictions(y)
    df = tweet_features(texts, data["train"][0], emb)
    df.insert(0, "label", y)
    for s, p in preds.items():
        df[f"pred {s}"] = p
    df.to_csv(RESULTS / "step6_features.csv.gz", index_label="id")
    print(df[FEATURES].describe().round(3).to_string())

    table = binned_scores(df, preds, y)
    (RESULTS / "step6_bins.json").write_text(table.to_json(orient="records", indent=1))
    plot_bins(table, preds)
    print(table.round(3).to_string())

    probs_rnn = np.load(RESULTS / "step5_cv" / setting_name(cv_candidates()["best"])
                        / "test_ensemble_probs.npy").astype(np.float32)
    ex = per_class_and_examples(texts, df, preds, y, probs_rnn)
    (RESULTS / "step6_examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False))
    print(pd.DataFrame(ex["per_class_f1"]).sort_values("RNN - pool").to_string())
    print(json.dumps(ex["disagreement"], indent=1))

    bda = run_bda(df, preds, y)
    (RESULTS / "step6_bda.json").write_text(json.dumps(bda, indent=1))
    print(json.dumps({k: bda[k] for k in ("divergences", "max_rhat", "accuracy")}, indent=1))
    for f in FEATURES:
        print(f"{f:<13}", "  ".join(f"{s}: {v['mean']:+.3f} (P>0 {v['P(>0)']:.2f})" for s, v in bda["weights"][f].items()))


if __name__ == "__main__":
    main()
