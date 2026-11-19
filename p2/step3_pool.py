"""
Step 3: embed-and-pool classifier.

    tweet -> tokens -> frozen pretrained word vectors -> pool (mean / max / both) -> classifier -> 20 emojis

Two phases:
  sweep : train on `train`, score on `validation`, 3 seeds per setting. One factor at a time is changed
          from the DEFAULT setting: embedding (GloVe 25/50/100/200, fastText 300), pooling, architecture,
          optimiser, class-weighted loss.            -> results/step3_sweep.json
  cv    : the best sweep setting, with Project 1's exact protocol (5 folds x seeds 0,1,2 on `train`),
          so every score pairs with a Project 1 score for the Bayesian comparison (step 5). Every fold model
          also predicts the 50k test tweets; we save those plus per-tweet features for error analysis.
                                                      -> results/step3_cv/

Run:  .venv/bin/python -m p2.step3_pool sweep
      .venv/bin/python -m p2.step3_pool cv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from p2.common import NAMES, RESULTS, kfold_ids, load_data, load_embeddings, metrics

DEFAULT = {"emb": "glove200", "pool": "meanmax", "arch": "mlp", "opt": "adam", "weighted": True}
SWEEP = {
    "emb": ["glove25", "glove50", "glove100", "glove200", "fasttext300", "glovewiki200", "word2vec300"],
    "pool": ["mean", "max", "meanmax"],
    "arch": ["linear", "mlp", "mlp2"],
    "opt": ["adam_lr1e-4", "adam", "adam_lr1e-2", "sgd_lr0.01", "sgd", "sgd_momentum"],   # optimiser + learning rate
    "weighted": [True, False],
}

# ------------------------------------------------------------------------------------------
# Embed and pool (embeddings are frozen in step 3, so pooled vectors are computed once)
# ------------------------------------------------------------------------------------------

def pool_texts(texts, emb, pool):
    out = np.zeros((len(texts), emb.dim * (2 if pool == "meanmax" else 1)), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [emb[w] for w in emb.tokenize(t) if w in emb]   # out-of-vocabulary tokens are skipped
        if not vecs:
            continue                                          # no known word: all-zero vector
        V = np.stack(vecs)
        out[i] = {"mean": V.mean(0), "max": V.max(0),
                  "meanmax": np.concatenate([V.mean(0), V.max(0)])}[pool]
    return out


def tweet_stats(texts, emb):
    toks = [emb.tokenize(t) for t in texts]
    n = np.array([len(t) for t in toks])
    n_oov = np.array([sum(w not in emb for w in t) for t in toks])
    return pd.DataFrame({"n_tokens": n, "n_oov": n_oov, "oov_rate": n_oov / np.maximum(n, 1),
                         "n_chars": [len(t) for t in texts]})

# ------------------------------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------------------------------

def make_net(arch, d_in, n_classes=20):
    if arch == "linear":                                       # = multinomial logistic regression
        return nn.Linear(d_in, n_classes)
    if arch == "mlp":
        return nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
    if arch == "mlp2":
        return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.3),
                             nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))


def make_opt(opt, params):
    return {"adam": lambda: torch.optim.Adam(params, lr=1e-3),
            "adam_lr1e-4": lambda: torch.optim.Adam(params, lr=1e-4),
            "adam_lr1e-2": lambda: torch.optim.Adam(params, lr=1e-2),
            "sgd": lambda: torch.optim.SGD(params, lr=0.05),
            "sgd_lr0.01": lambda: torch.optim.SGD(params, lr=0.01),
            "sgd_momentum": lambda: torch.optim.SGD(params, lr=0.05, momentum=0.9)}[opt]()


class PooledClassifier:
    """
    Standardises the pooled vectors, then trains `arch` with cross-entropy (optionally weighted by inverse
    class frequency, which helps macro-F1 on our imbalanced labels). Like Project 1, 10% of the training data
    is held out to pick the best epoch (by macro-F1); the scored data is never used for that.
    """

    def __init__(self, arch, opt, weighted, seed, max_epochs=30, patience=4, batch_size=128):
        self.arch, self.opt, self.weighted, self.seed = arch, opt, weighted, seed
        self.max_epochs, self.patience, self.bs = max_epochs, patience, batch_size

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = torch.tensor((X - self.mu) / self.sd)
        y = torch.tensor(y)
        perm = rng.permutation(len(y))
        dev, tr = perm[: len(y) // 10], perm[len(y) // 10:]

        self.net = make_net(self.arch, X.shape[1])
        opt = make_opt(self.opt, self.net.parameters())
        counts = torch.bincount(y[tr], minlength=20).float()
        weight = (counts.sum() / (20 * counts)) if self.weighted else None
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = tr[rng.permutation(len(tr))]
            for s in range(0, len(order), self.bs):
                b = order[s:s + self.bs]
                opt.zero_grad()
                loss_fn(self.net(X[b]), y[b]).backward()
                opt.step()
            f1 = metrics(y[dev].numpy(), self._predict(X[dev]))["macro_f1"]
            if f1 > best:
                best, bad, self.best_epoch = f1, 0, epoch
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def _predict(self, Xt):
        self.net.eval()
        with torch.no_grad():
            return self.net(Xt).argmax(1).numpy()

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor((X - self.mu) / self.sd)), 1).numpy()

# ------------------------------------------------------------------------------------------
# Phases
# ------------------------------------------------------------------------------------------

def sweep_settings():
    seen, out = set(), []
    for factor, values in SWEEP.items():
        for v in values:
            s = {**DEFAULT, factor: v}
            key = json.dumps(s, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append((factor, s))
    return out


def run_sweep(data, seeds=(0, 1, 2)):
    cache, rows = {}, []
    for factor, s in sweep_settings():
        if s["emb"] not in cache:
            cache[s["emb"]] = load_embeddings(s["emb"])
        emb = cache[s["emb"]]
        Xtr = pool_texts(data["train"][0], emb, s["pool"])
        Xva = pool_texts(data["validation"][0], emb, s["pool"])
        for seed in seeds:
            clf = PooledClassifier(s["arch"], s["opt"], s["weighted"], seed).fit(Xtr, data["train"][1])
            m = metrics(data["validation"][1], clf.predict_proba(Xva).argmax(1))
            rows.append({"factor": factor, **s, "seed": seed, **m, "best_epoch": clf.best_epoch})
            print(f"{factor:<9} {json.dumps(s)} seed={seed}  val macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")
    (RESULTS / "step3_sweep.json").write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    summary = (df.groupby(["factor", "emb", "pool", "arch", "opt", "weighted"])[["macro_f1", "accuracy"]]
               .agg(["mean", "std"]).round(4))
    print(summary.to_string())


def best_setting():
    df = pd.DataFrame(json.loads((RESULTS / "step3_sweep.json").read_text()))
    keys = list(DEFAULT)
    best = df.groupby(keys).macro_f1.mean().idxmax()
    return dict(zip(keys, [bool(v) if isinstance(v, (bool, np.bool_)) else v for v in best]))


def run_cv(data, seeds=(0, 1, 2), k=5):
    s = best_setting()
    name = f"{s['emb']}-{s['pool']}"
    model = f"{s['arch']}-{s['opt']}{'-weighted' if s['weighted'] else ''}"
    print("chosen on validation:", s)
    out = RESULTS / "step3_cv"
    (out / "test_preds").mkdir(parents=True, exist_ok=True)

    emb = load_embeddings(s["emb"])
    X = pool_texts(data["train"][0], emb, s["pool"])
    X_test = pool_texts(data["test"][0], emb, s["pool"])
    y, y_test = data["train"][1], data["test"][1]

    rows, test_probs = [], []
    for seed in seeds:
        folds = kfold_ids(len(y), k, seed)
        for f in range(k):
            tr = np.concatenate([folds[j] for j in range(k) if j != f])
            clf = PooledClassifier(s["arch"], s["opt"], s["weighted"], seed).fit(X[tr], y[tr])
            m = metrics(y[folds[f]], clf.predict_proba(X[folds[f]]).argmax(1))
            rows.append({"encoder": name, "model": model, "system": f"{model} | {name}", "seed": seed, "fold": f,
                         **m, "best_epoch": clf.best_epoch})
            p = clf.predict_proba(X_test)
            test_probs.append(p)
            np.save(out / "test_preds" / f"s{seed}_f{f}.npy", p.argmax(1).astype(np.int8))
            print(f"seed {seed} fold {f}: CV macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}")

    (out / "cv_results.json").write_text(json.dumps(rows, indent=1))
    np.save(out / "y_test.npy", y_test.astype(np.int8))

    # test set: every single fold model, and the ensemble (average of the 15 models' probabilities)
    single = [metrics(y_test, p.argmax(1)) for p in test_probs]
    ens_prob = np.mean(test_probs, axis=0)
    np.save(out / "test_ensemble_probs.npy", ens_prob.astype(np.float16))
    test = {"setting": s, "system": f"{model} | {name}",
            "cv_macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
            "cv_macro_f1_std": float(np.std([r["macro_f1"] for r in rows], ddof=1)),
            "test_single_macro_f1_mean": float(np.mean([m["macro_f1"] for m in single])),
            "test_single_macro_f1_std": float(np.std([m["macro_f1"] for m in single], ddof=1)),
            "test_single_accuracy_mean": float(np.mean([m["accuracy"] for m in single])),
            "test_ensemble": metrics(y_test, ens_prob.argmax(1)),
            # majority vote of the 15 models: the ensemble Project 1 reported (0.195 macro-F1)
            "test_ensemble_vote": metrics(y_test, np.array([np.bincount(c, minlength=20).argmax()
                                                             for c in np.stack([p.argmax(1) for p in test_probs]).T])),
            # which emojis are hard (used on slide 2 as evidence for the limitations of the embeddings)
            "per_class_f1_ensemble": dict(zip(NAMES, np.round(f1_score(y_test, ens_prob.argmax(1), average=None), 3)
                                              .tolist()))}
    (out / "test_summary.json").write_text(json.dumps(test, indent=1))
    print(json.dumps(test, indent=1))

    # one row per test tweet: features for error analysis (step 6)
    ex = tweet_stats(data["test"][0], emb)
    ex.insert(0, "text", data["test"][0])
    ex.insert(1, "label", y_test)
    ex["pred_ensemble"] = ens_prob.argmax(1)
    ex["correct"] = ex.pred_ensemble == ex.label
    ex["p_label"] = ens_prob[np.arange(len(y_test)), y_test]
    ex.to_csv(out / "test_examples.csv", index_label="id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sweep", "cv"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    data = load_data()
    run_sweep(data) if args.phase == "sweep" else run_cv(data)


if __name__ == "__main__":
    main()
