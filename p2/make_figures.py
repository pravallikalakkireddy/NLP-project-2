"""
Slide figures for steps 4-6, in the same style as slides 1-3 (blue = our model, grey = comparison).
Reads only saved results, so it takes a few seconds.

Run:  .venv/bin/python -m p2.make_figures        -> results/figures/slide4_*.png, slide5_*.png, slide6_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p2.common import RESULTS
from p2.step4_rnn import DEFAULT, SWEEP, load_rows, setting_name
from p2.step5_search import cv_candidates

FIG = RESULTS / "figures"
BLUE, GREY, ORANGE, VIOLET = "#2a78d6", "#9a9a94", "#eb6834", "#4a3aa7"
SYSTEM_COLORS = {"P1 BoW": GREY, "P2 pool": ORANGE, "P2 RNN": BLUE}
SYSTEM_LABELS = {"P1 BoW": "Project 1: logistic · BoW+bigrams", "P2 pool": "step 3: GloVe · mean+max · MLP",
                 "P2 RNN": "step 4: GloVe · BiGRU"}
POOL_VAL = 20.7                     # step 3 model, validation macro-F1 (3 seeds)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})


def bar_labels(ax, bars, fmt="{:.1f}", size=7):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=size)


def slide4_sweep():
    """Step 4 sweep: change one factor of the default GRU, 3 seeds each (mean ± 1 sd)."""
    df = pd.DataFrame(load_rows(RESULTS / "step4_sweep.json"))
    default = df.name == setting_name(DEFAULT)
    labels = {"cell": {"gru": "GRU", "lstm": "LSTM"},
              "emb": {"frozen": "frozen", "trainable": "train-\nable", "random": "random"},
              "bidir": {False: "one\nway", True: "both\nways"},
              "readout": {"last": "last", "max": "max", "meanmax": "mean\n+max"},
              "word_drop": {0.0: "0", 0.25: "0.25"}}
    titles = {"cell": "cell", "emb": "embeddings", "bidir": "direction", "readout": "readout", "word_drop": "word dropout"}
    fig, axes = plt.subplots(1, len(SWEEP), figsize=(7.4, 2.7), sharey=True,
                             gridspec_kw={"width_ratios": [len(v) for v in SWEEP.values()]})
    for ax, (factor, values) in zip(axes, SWEEP.items()):
        stats = [df[(default | (df.tag == factor)) & (df[factor] == v)].macro_f1 * 100 for v in values]
        means = [s.mean() for s in stats]
        colors = [BLUE if m == max(means) else GREY for m in means]
        sds = [s.std() for s in stats]
        ax.bar(range(len(values)), means, 0.7, yerr=sds, color=colors,
               error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#52514e"})
        for xi, (m, sd) in enumerate(zip(means, sds)):          # label above the error bar, not on it
            ax.annotate(f"{m:.1f}", (xi, m + sd), xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7)
        ax.axhline(POOL_VAL, color=ORANGE, lw=1, ls="--")
        ax.set_xticks(range(len(values)), [labels[factor][v] for v in values], fontsize=7.5)
        ax.set_title(titles[factor], fontsize=9)
        ax.set_ylim(0, 26)
    axes[0].set_ylabel("validation macro-F1 (%)")
    fig.suptitle("Change one factor of the default GRU (frozen GloVe 200d · 128 units · last state) · 3 seeds, ±1 sd\n"
                 "blue = best per factor · dashed = step 3 embed-and-pool (20.7)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "slide4_sweep.png", dpi=200)


def slide4_datasize():
    df = pd.DataFrame(load_rows(RESULTS / "step4_datasize.json"))
    series = [("GRU · frozen GloVe", (df.tag == "datasize") & (df.emb == "frozen"), BLUE, "o", "-"),
              ("GRU · trainable GloVe", (df.tag == "datasize") & (df.emb == "trainable"), ORANGE, "s", "-"),
              ("GRU · random vectors", (df.tag == "datasize") & (df.emb == "random"), VIOLET, "^", "-"),
              ("embed-and-pool (step 3)", df.tag == "pool", GREY, "D", "--")]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for name, m, color, marker, ls in series:
        g = df[m].groupby("frac").macro_f1
        x, y, sd = g.mean().index * 100, g.mean() * 100, g.std() * 100
        ax.errorbar(x, y, yerr=sd, color=color, marker=marker, ms=6, lw=2, ls=ls, capsize=2, label=name)
        ax.annotate(f"{name}  {y.iloc[-1]:.1f}", (x[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color="#0b0b0b")
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100], ["10%\n(4.5k)", "30%\n(13.5k)", "100%\n(45k tweets)"])
    ax.set_xlim(8, 420)
    ax.spines["bottom"].set_bounds(10, 100)
    ax.set_ylabel("validation macro-F1 (%)")
    ax.set_title("Training data size · 3 seeds, ±1 sd", fontsize=9)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "slide4_datasize.png", dpi=200)


def slide5_cv_test():
    """Same layout as slide 3's chart, with the GRU added."""
    p1 = [r["macro_f1"] for r in json.loads((RESULTS / "p1_cv_results.json").read_text())
          if r["system"] == "logistic | bow+bigrams"]
    pool = json.loads((RESULTS / "step3_cv" / "test_summary.json").read_text())
    rnn = json.loads((RESULTS / "step5_cv" / setting_name(cv_candidates()["best"]) / "test_summary.json").read_text())
    from p2.common import metrics
    y_test = np.load(RESULTS / "step3_cv" / "y_test.npy")
    p1_test = np.stack([np.load(RESULTS / "p1_test_preds" / f"bow+bigrams__logistic__s{s}__f{f}.npy")
                        for s in range(3) for f in range(5)])
    p1_vote = np.array([np.bincount(c, minlength=20).argmax() for c in p1_test.T])
    values = {"P1 BoW": [np.mean(p1) * 100, np.mean([metrics(y_test, p)["macro_f1"] for p in p1_test]) * 100,
                         metrics(y_test, p1_vote)["macro_f1"] * 100],
              "P2 pool": [pool["cv_macro_f1_mean"] * 100, pool["test_single_macro_f1_mean"] * 100,
                          pool["test_ensemble_vote"]["macro_f1"] * 100],
              "P2 RNN": [rnn["cv_macro_f1_mean"] * 100, rnn["test_single_macro_f1_mean"] * 100,
                         rnn["test_ensemble_vote"]["macro_f1"] * 100]}     # majority vote for all three
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = np.arange(3)
    for i, (s, v) in enumerate(values.items()):
        bars = ax.bar(x + (i - 1) * 0.27, v, 0.25, color=SYSTEM_COLORS[s], label=SYSTEM_LABELS[s])
        bar_labels(ax, bars, size=7.5)
    ax.set_xticks(x, ["CV (5 folds × 3 seeds)", "test, single models", "test, majority vote of 15"])
    ax.set_ylabel("macro-F1 (%)")
    ax.set_ylim(0, 29)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "slide5_cv_test.png", dpi=200)


def slide5_search():
    """Backup slide: every RNN setting tried on validation, 3 seeds each."""
    rows = load_rows(RESULTS / "step4_sweep.json") + load_rows(RESULTS / "step5_search.json")
    df = pd.DataFrame([r for r in rows if r["frac"] == 1.0])
    order = df.groupby("name").macro_f1.mean().sort_values().index
    best = order[-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, name in enumerate(order):
        v = df[df.name == name].macro_f1 * 100
        color = BLUE if name == best else GREY
        ax.plot([v.min(), v.max()], [i, i], color=color, lw=1)
        ax.plot(v, [i] * len(v), "o", color=color, ms=4.5, mec="white", mew=0.8)
        ax.plot(v.mean(), i, "|", color="#0b0b0b", ms=11, mew=2)
    pretty = [n.replace("-h128", "").replace("-lr0.001", "").replace("-wd0.0", "").replace("-wd", " · drop ")
              .replace("-bi", " · bi").replace("-", " · ") for n in order]
    src = df.groupby("name").tag.first()
    ax.set_yticks(range(len(order)), [f"{p}   ({'search' if src[n] == 'search' else 'sweep'})"
                                      for p, n in zip(pretty, order)], fontsize=7)
    ax.axvline(POOL_VAL, color=ORANGE, lw=1, ls="--")
    ax.set_xlabel("validation macro-F1 (%)\ndots = 3 seeds · bar = mean · dashed = embed-and-pool (step 3)", fontsize=8)
    ax.grid(axis="x", color="#e6e6e3", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "slide5_search.png", dpi=200)


def slide6_bins():
    t = pd.DataFrame(json.loads((RESULTS / "step6_bins.json").read_text()))
    panels = [("unseen_rate", "words never seen in training"), ("n_tokens", "tweet length (tokens)"),
              ("negation", "contains a negation")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [4, 5, 2]})
    for ax, (feat, title) in zip(axes, panels):
        d = t[t.feature == feat]
        x = np.arange(len(d))
        for i, s in enumerate(SYSTEM_COLORS):
            ax.bar(x + (i - 1) * 0.27, d[f"{s} macro_f1"] * 100, 0.25, color=SYSTEM_COLORS[s],
                   label=SYSTEM_LABELS[s])
        for xi, v in zip(x, d["P2 RNN macro_f1"] * 100):           # label our model only (selective labels)
            ax.annotate(f"{v:.1f}", (xi + 0.27, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=7)
        for xi, v in zip(x, d["P1 BoW macro_f1"] * 100):
            ax.annotate(f"{v:.1f}", (xi - 0.27, v), xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=7, color="#52514e")
        ax.set_xticks(x, [f"{b}\n{n:,}" for b, n in zip(d.bin, d.n)], fontsize=6.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 28)
    axes[0].set_ylabel("test macro-F1 (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "slide6_bins.png", dpi=200)


def slide6_bda():
    b = json.loads((RESULTS / "step6_bda.json").read_text())
    feats = list(b["weights"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, s in enumerate(SYSTEM_COLORS):
        for j, f in enumerate(feats):
            w = b["weights"][f][s]
            y = j + (i - 1) * 0.24
            ax.plot(w["hdi95"], [y, y], color=SYSTEM_COLORS[s], lw=2, solid_capstyle="round")
            ax.plot(w["mean"], y, "o", color=SYSTEM_COLORS[s], ms=5, mec="white", mew=0.8,
                    label=SYSTEM_LABELS[s] if j == 0 else None)
    ax.axvline(0, color="#52514e", lw=0.7)
    ax.set_yticks(range(len(feats)), feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("effect on log-odds of a correct prediction\n(standardised feature, 95% interval)", fontsize=8)
    fig.suptitle(f"Bayesian correctness model · 10k test tweets · r̂ ≤ {b['max_rhat']:.3f}, "
                 f"{b['divergences']} divergences", fontsize=8)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6.5, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.95), handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="x", color="#e6e6e3", lw=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "slide6_bda.png", dpi=200)


def main():
    for f in (slide4_sweep, slide4_datasize, slide5_cv_test, slide5_search, slide6_bins, slide6_bda):
        f()
        print("saved", f.__name__)


if __name__ == "__main__":
    main()
