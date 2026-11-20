"""
Step 5: decisions under uncertainty: hyperparameter search, experimental variance, Bayesian comparisons.

How the search is structured (coarse to fine):
  1. step 3 / step 4 sweeps: one factor at a time from a default, 3 seeds each, on the validation set.
     Cheap, shows the main effect of every factor, but never tests combinations.
  2. search (this file): random search over COMBINATIONS of the values that were competitive in step 4
     (within 1 point of the best value of their factor), 6 settings x 3 seeds, on the validation set.
                                                                         -> results/step5_search.json
  3. cv (this file): the best RNN setting and its frozen <-> trainable counterpart, with Project 1's
     protocol (5 folds x seeds 0,1,2 on `train`). Every fold model also predicts the 50k test tweets.
                                                                         -> results/step5_cv/<setting>/
  4. compare (this file): Bayesian correlated t-test with a ROPE (Benavoli et al., course part1/analysis.py)
     on the paired CV scores, plus a variance breakdown (seed vs fold).  -> results/step5_compare.json, figures

Selecting the maximum of many noisy validation scores is optimistic (winner's curse); that is why the chosen
settings are re-estimated with fresh folds and seeds in phase 3 before we compare systems.

Run:  .venv/bin/python -m p2.step5_search search
      .venv/bin/python -m p2.step5_search cv
      .venv/bin/python -m p2.step5_search compare
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from p2.common import NAMES, RESULTS, kfold_ids, metrics
from p2.step4_rnn import (DEFAULT, SWEEP, RNNClassifier, best_sweep_setting, load_rows, prepare, run_settings,
                          setting_name)

FIG = RESULTS / "figures"
ROPE = 0.01                     # 1 macro-F1 point, as in Project 1

N_SEARCH = 6
KEEP_WITHIN = 0.01              # a value stays in the search if it is within 1 point of its factor's best

# ------------------------------------------------------------------------------------------
# 2. Random search
# ------------------------------------------------------------------------------------------

def search_space():
    """
    Coarse to fine: from the step 4 sweep, keep for every factor the values whose mean validation macro-F1 is
    within KEEP_WITHIN of that factor's best value (differences smaller than that are seed noise).
    """
    df = pd.DataFrame(load_rows(RESULTS / "step4_sweep.json"))
    default_runs = df.name == setting_name(DEFAULT)          # the default is stored once, under the first factor
    space = {}
    for factor, values in SWEEP.items():
        means = {v: df[(default_runs | (df.tag == factor)) & (df[factor] == v)].macro_f1.mean() for v in values}
        top = max(means.values())
        space[factor] = [v for v in values if means[v] >= top - KEEP_WITHIN]
    return space


def search_settings(n=N_SEARCH, seed=0):
    space = search_space()
    rng = np.random.default_rng(seed)
    seen = {setting_name({**DEFAULT, f: v}) for f, vals in SWEEP.items() for v in vals}   # already run in step 4
    n = min(n, int(np.prod([len(v) for v in space.values()])) - 1)
    out, tries = [], 0
    while len(out) < n and tries < 10_000:
        tries += 1
        s = {**DEFAULT, **{k: v[rng.integers(len(v))] for k, v in space.items()}}
        s = {k: (bool(v) if k == "bidir" else v.item() if hasattr(v, "item") else v) for k, v in s.items()}
        if setting_name(s) not in seen:
            seen.add(setting_name(s))
            out.append(s)
    return out


def all_validation_rows():
    """Every full-data validation run of steps 4 and 5 (same seeds, same validation set)."""
    rows = load_rows(RESULTS / "step4_sweep.json") + load_rows(RESULTS / "step5_search.json")
    return pd.DataFrame([r for r in rows if r["frac"] == 1.0])


def best_rnn_setting():
    df = all_validation_rows()
    name = df.groupby("name").macro_f1.mean().idxmax()
    r = df[df.name == name].iloc[0]
    return {k: (bool(r[k]) if k == "bidir" else r[k].item() if hasattr(r[k], "item") else r[k]) for k in DEFAULT}


def run_search(data, vocab, seqs, seeds=(0, 1, 2)):
    path = RESULTS / "step5_search.json"
    print("search space (step 4 values within 1 point of their factor's best):", search_space())
    jobs = [("search", s, seed, 1.0) for s in search_settings() for seed in seeds]
    run_settings(load_rows(path), path, jobs, data, vocab, seqs)

    df = all_validation_rows()
    g = df.groupby("name").macro_f1
    table = pd.DataFrame({"mean": g.mean(), "sd_seeds": g.std(), "min": g.min(), "max": g.max(),
                          "source": df.groupby("name").tag.first()}).sort_values("mean")
    print(table.round(4).to_string())
    print("best RNN setting:", best_rnn_setting())

    # every setting: mean and the 3 seeds; is the spread across seeds as big as the gaps between settings?
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(table) + 1.2))
    for i, name in enumerate(table.index):
        vals = df[df.name == name].macro_f1 * 100
        ax.plot(vals, [i] * len(vals), "o", color="#9aa5b1", ms=4)
        ax.plot(vals.mean(), i, "|", color="#1f4e79", ms=14, mew=2.5)
    ax.set_yticks(range(len(table)), [f"{n}  ({s})" for n, s in zip(table.index, table.source)], fontsize=7)
    ax.set_xlabel("validation macro-F1 (%)   grey dots = 3 seeds, bar = mean")
    ax.set_title("Step 4 sweep + step 5 random search: every RNN setting")
    fig.tight_layout()
    fig.savefig(FIG / "step5_search.png", dpi=200)

# ------------------------------------------------------------------------------------------
# 3. Final systems: 5 folds x 3 seeds (Project 1's folds), plus test predictions
# ------------------------------------------------------------------------------------------

def cv_candidates():
    best = best_rnn_setting()
    # the twin with the other embedding setting (random embeddings are covered by step 4's data-size runs)
    twin = "trainable" if best["emb"] == "frozen" else "frozen"
    return {"best": best, twin: {**best, "emb": twin}}


def add_row(path, row):
    """Append one result under a file lock, so several `cv --seeds` processes can run in parallel."""
    import fcntl
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [r for r in load_rows(path) if (r["seed"], r["fold"]) != (row["seed"], row["fold"])] + [row]
        path.write_text(json.dumps(rows, indent=1))
    return rows


def run_cv(data, vocab, seqs, emb, seeds=(0, 1, 2), k=5, roles=None):
    from p2.step3_pool import tweet_stats
    y, y_test = data["train"][1], data["test"][1]
    for role, s in cv_candidates().items():
        if roles and role not in roles:
            continue
        name = setting_name(s)
        out = RESULTS / "step5_cv" / name
        (out / "test_probs").mkdir(parents=True, exist_ok=True)
        rows = load_rows(out / "cv_results.json")
        print(f"== {role}: {name}", flush=True)
        for seed in seeds:
            folds = kfold_ids(len(y), k, seed)
            for f in range(k):
                prob_path = out / "test_probs" / f"s{seed}_f{f}.npy"
                if any(r["seed"] == seed and r["fold"] == f for r in rows) and prob_path.exists():
                    continue
                tr = np.concatenate([folds[j] for j in range(k) if j != f])
                clf = RNNClassifier(vocab, s, seed).fit([seqs["train"][i] for i in tr], y[tr])
                m = metrics(y[folds[f]], clf.predict_proba([seqs["train"][i] for i in folds[f]]).argmax(1))
                np.save(prob_path, clf.predict_proba(seqs["test"]).astype(np.float16))
                rows = add_row(out / "cv_results.json",
                               {"encoder": "glove200-" + name.split("-", 2)[2], "model": s["cell"], "system": name,
                                "role": role, "seed": seed, "fold": f, **m, "best_epoch": clf.best_epoch})
                print(f"  seed {seed} fold {f}: CV macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}"
                      f"  epoch {clf.best_epoch}", flush=True)

        rows = load_rows(out / "cv_results.json")
        if len(rows) < len(seeds) * k or len(seeds) < 3:
            print(f"  {len(rows)} of 15 CV models done; summary is written when all are in", flush=True)
            continue
        probs = [np.load(out / "test_probs" / f"s{r['seed']}_f{r['fold']}.npy").astype(np.float32)
                 for r in sorted(rows, key=lambda r: (r["seed"], r["fold"]))]
        single = [metrics(y_test, p.argmax(1)) for p in probs]
        ens = np.mean(probs, axis=0)
        np.save(out / "test_ensemble_probs.npy", ens.astype(np.float16))
        np.save(out / "y_test.npy", y_test.astype(np.int8))
        summary = {"setting": s, "role": role, "system": name,
                   "cv_macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
                   "cv_macro_f1_std": float(np.std([r["macro_f1"] for r in rows], ddof=1)),
                   "test_single_macro_f1_mean": float(np.mean([m["macro_f1"] for m in single])),
                   "test_single_macro_f1_std": float(np.std([m["macro_f1"] for m in single], ddof=1)),
                   "test_single_accuracy_mean": float(np.mean([m["accuracy"] for m in single])),
                   "test_ensemble": metrics(y_test, ens.argmax(1)),
                   "per_class_f1_ensemble": dict(zip(NAMES, np.round(
                       f1_score(y_test, ens.argmax(1), average=None), 3).tolist()))}
        (out / "test_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if k != "per_class_f1_ensemble"}, indent=1))

        ex = tweet_stats(data["test"][0], emb)
        ex.insert(0, "text", data["test"][0])
        ex.insert(1, "label", y_test)
        ex["pred_ensemble"] = ens.argmax(1)
        ex["correct"] = ex.pred_ensemble == ex.label
        ex["p_label"] = ens[np.arange(len(y_test)), y_test]
        ex.to_csv(out / "test_examples.csv", index_label="id")

# ------------------------------------------------------------------------------------------
# 4. Bayesian comparisons on the paired CV scores + variance breakdown
# ------------------------------------------------------------------------------------------

def cv_scores():
    """system label -> 15 macro-F1 scores sorted by (seed, fold), so systems pair up fold by fold."""
    def sorted_scores(rows):
        return np.array([r["macro_f1"] for r in sorted(rows, key=lambda r: (r["seed"], r["fold"]))])

    p1 = json.loads((RESULTS / "p1_cv_results.json").read_text())
    out = {"P1 BoW+bigrams logistic": sorted_scores([r for r in p1 if r["system"] == "logistic | bow+bigrams"]),
           "P2 embed-and-pool": sorted_scores(json.loads((RESULTS / "step3_cv" / "cv_results.json").read_text()))}
    for role, s in cv_candidates().items():
        path = RESULTS / "step5_cv" / setting_name(s) / "cv_results.json"
        if path.exists():
            label = f"RNN {s['emb']}" + (" (best)" if role == "best" else "")
            out[label] = sorted_scores(json.loads(path.read_text()))
    return out


def variance_breakdown(scores, runs=3, k=5):
    """How much of the spread comes from the seed (different folds + init) vs the fold within a seed."""
    out = {}
    for name, x in scores.items():
        m = x.reshape(runs, k)
        out[name] = {"mean": float(x.mean()), "sd_all_15": float(x.std(ddof=1)),
                     "sd_between_seed_means": float(m.mean(1).std(ddof=1)),
                     "sd_within_seed_folds": float(np.sqrt(m.var(1, ddof=1).mean()))}
    return out


def run_compare():
    from part1.analysis import BenavoliCorrelatedT

    scores = cv_scores()
    names = list(scores)
    rnn_best = next(n for n in names if n.endswith("(best)"))
    pairs = [("P1 BoW+bigrams logistic", "P2 embed-and-pool"),
             ("P2 embed-and-pool", rnn_best),
             ("P1 BoW+bigrams logistic", rnn_best)]
    pairs += [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
              if a.startswith("RNN") and b.startswith("RNN")]          # frozen vs trainable vs random

    results = []
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 3.2))
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        t = BenavoliCorrelatedT(scores[a], scores[b], runs=3, names=[a, b])
        row = {"A": a, "B": b, "mean_A": float(scores[a].mean()), "mean_B": float(scores[b].mean()),
               "mean_diff_B_minus_A": float(t.mean_diff)}
        for rope in (0.005, ROPE, 0.02):                 # main ROPE 0.01, plus a sensitivity check
            pa, pt, pb = t.probs(rope)
            row[f"rope{rope}"] = {"P(A better)": float(pa), "P(equal)": float(pt), "P(B better)": float(pb)}
        results.append(row)
        # posterior of the difference B - A (Student t), with the ROPE shaded
        xs = np.linspace(t.mean_diff - 5 * t.std, t.mean_diff + 5 * t.std, 400)
        from scipy import stats
        ax.plot(xs * 100, stats.t.pdf(xs, t.df, t.mean_diff, t.std), color="#1f4e79")
        ax.axvspan(-ROPE * 100, ROPE * 100, color="#f2c14e", alpha=0.35, label="ROPE ±1 pt")
        ax.axvline(0, color="grey", lw=0.6)
        p = row[f"rope{ROPE}"]
        ax.set_title(f"{b}\nminus {a}", fontsize=8)
        ax.set_xlabel(f"Δ macro-F1 (pts)\nP(A)={p['P(A better)']:.2f}  P(=)={p['P(equal)']:.2f}"
                      f"  P(B)={p['P(B better)']:.2f}", fontsize=7)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "step5_bayes.png", dpi=200)

    # box plot of the 15 CV scores per system
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.boxplot([scores[n] * 100 for n in names], vert=False, widths=0.5)
    for i, n in enumerate(names):
        ax.plot(scores[n] * 100, np.full(15, i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, 15), "o",
                ms=3, alpha=0.6, color="#1f4e79")
    ax.set_yticks(range(1, len(names) + 1), names, fontsize=8)
    ax.set_xlabel("cross-validation macro-F1 (%), 5 folds × 3 seeds")
    fig.tight_layout()
    fig.savefig(FIG / "step5_cv_box.png", dpi=200)

    out = {"rope": ROPE, "comparisons": results, "variance": variance_breakdown(scores)}
    (RESULTS / "step5_compare.json").write_text(json.dumps(out, indent=1))
    for r in results:
        p = r[f"rope{ROPE}"]
        print(f"{r['A']:<26} vs {r['B']:<26} diff(B-A)={r['mean_diff_B_minus_A'] * 100:+.2f} pts   "
              f"P(A)={p['P(A better)']:.3f} P(=)={p['P(equal)']:.3f} P(B)={p['P(B better)']:.3f}")
    print(pd.DataFrame(out["variance"]).T.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["search", "cv", "compare"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="cv: run only these seeds (parallel runs)")
    ap.add_argument("--roles", nargs="+", help="cv: run only these systems (best / frozen / trainable)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.phase == "compare":
        return run_compare()
    import torch
    torch.set_num_threads(args.threads)
    data, emb, vocab, seqs = prepare()
    if args.phase == "search":
        run_search(data, vocab, seqs)
    else:
        run_cv(data, vocab, seqs, emb, seeds=tuple(args.seeds), roles=args.roles)


if __name__ == "__main__":
    main()
