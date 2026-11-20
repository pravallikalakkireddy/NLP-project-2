"""
Step 4: recurrent encoder (GRU / LSTM) on top of GloVe Twitter 200d.

    tweet -> GloVe tokens -> word vectors (frozen / trainable / random) -> GRU or LSTM -> readout -> MLP -> 20 emojis

Only the encoder changes compared with step 3: same tokeniser, same embeddings, same classifier head
(MLP 256 + dropout 0.3), same class-weighted loss, same "best epoch on 10% of the training data" rule.

Phases:
  sweep    : train on `train`, score on `validation`, 3 seeds per setting. One factor at a time is changed
             from DEFAULT: cell, embeddings (frozen / trainable / random), direction, readout, word dropout.
                                                                        -> results/step4_sweep.json
  datasize : 10% / 30% / 100% of the training tweets, for frozen / trainable / random embeddings
             and for the step 3 embed-and-pool model, 3 seeds each, scored on validation.
                                                                        -> results/step4_datasize.json

The final 5 folds x 3 seeds runs (and the Bayesian comparisons) are in step 5, which reuses `RNNClassifier`.

Run:  .venv/bin/python -m p2.step4_rnn sweep
      .venv/bin/python -m p2.step4_rnn datasize
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from p2.common import RESULTS, load_data, load_embeddings, metrics

PAD, UNK = 0, 1
MAX_LEN = 64             # longer tweets are cut (fewer than 0.1% of tweets have more GloVe tokens)

DEFAULT = {"cell": "gru", "emb": "frozen", "hidden": 128, "bidir": False, "readout": "last",
           "word_drop": 0.0, "lr": 1e-3}
SWEEP = {
    "cell": ["gru", "lstm"],
    "emb": ["frozen", "trainable", "random"],      # random = same vocabulary, no pretraining, trained end to end
    "bidir": [False, True],
    "readout": ["last", "max", "meanmax"],         # last hidden state, or max / mean+max over all time steps
    "word_drop": [0.0, 0.25],                      # zero out whole word vectors while training (course LSTMEncoder)
}
# hidden size (128) and learning rate (Adam 1e-3) stay fixed: step 3 already showed the learning-rate effect,
# and we spend the compute on the questions the project asks about recurrent encoders.

# ------------------------------------------------------------------------------------------
# Vocabulary: every GloVe word that occurs in any split. Row 0 = padding, row 1 = unknown word.
# Both are all-zero vectors, so a frozen model sees an unknown word as "something was here".
# Words missing from GloVe are UNK for every embedding setting, so all settings share one vocabulary.
# ------------------------------------------------------------------------------------------

class Vocab:
    def __init__(self, emb, texts_by_split):
        words = sorted({w for texts in texts_by_split for t in texts for w in emb.tokenize(t) if w in emb})
        self.words = ["<pad>", "<unk>"] + words
        self.index = {w: i for i, w in enumerate(self.words)}
        self.tokenize = emb.tokenize
        self.pretrained = np.zeros((len(self.words), emb.dim), dtype=np.float32)
        self.pretrained[2:] = np.stack([emb[w] for w in words])

    def encode(self, texts):
        out = []
        for t in texts:
            ids = [self.index.get(w, UNK) for w in self.tokenize(t)][:MAX_LEN]
            out.append(np.array(ids or [UNK], dtype=np.int64))
        return out

    def init_matrix(self, mode, seed):
        if mode in ("frozen", "trainable"):
            return self.pretrained.copy()
        # random: same size and scale as GloVe, but no pretrained knowledge
        rng = np.random.default_rng(1000 + seed)
        m = rng.normal(0, self.pretrained[2:].std(), self.pretrained.shape).astype(np.float32)
        m[:2] = 0
        return m


def pad_batch(seqs):
    lengths = torch.tensor([len(s) for s in seqs])
    x = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.from_numpy(s)
    return x, lengths

# ------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------

class RNNNet(nn.Module):
    def __init__(self, matrix, cell, hidden, bidir, readout, word_drop, freeze):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(matrix), freeze=freeze, padding_idx=PAD)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(matrix.shape[1], hidden, batch_first=True, bidirectional=bidir)
        self.cell, self.readout, self.word_drop = cell, readout, word_drop
        enc_dim = hidden * (2 if bidir else 1) * (2 if readout == "meanmax" else 1)
        self.head = nn.Sequential(nn.Linear(enc_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 20))

    def encode(self, x, lengths):
        e = self.embedding(x)
        valid = x.ne(PAD)
        if self.training and self.word_drop > 0:          # word dropout, as in the course's LSTMEncoder
            drop = (torch.rand(x.shape) < self.word_drop) & valid
            e = e.masked_fill(drop.unsqueeze(-1), 0.0)
        packed = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.readout == "last":
            h = h[0] if self.cell == "lstm" else h          # LSTM returns (hidden, cell state)
            return torch.cat(list(h), dim=1)                # (batch, hidden * directions)
        out, _ = pad_packed_sequence(out, batch_first=True)  # (batch, max_len, hidden * directions)
        mask = valid[:, :out.shape[1]].unsqueeze(-1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        if self.readout == "max":
            return mx
        mean = (out * mask).sum(1) / lengths.unsqueeze(-1)
        return torch.cat([mean, mx], dim=1)

    def forward(self, x, lengths):
        return self.head(self.encode(x, lengths))


class RNNClassifier:
    """
    Same training recipe as step 3's PooledClassifier: class-weighted cross-entropy, Adam, batch 128,
    best epoch chosen by macro-F1 on 10% of the training data (never on the data we report).
    Gradients are clipped at norm 1 (standard for RNNs, avoids exploding gradients).
    """

    def __init__(self, vocab, setting, seed, max_epochs=15, patience=3, batch_size=128):
        self.vocab, self.s, self.seed = vocab, setting, seed
        self.max_epochs, self.patience, self.bs = max_epochs, patience, batch_size

    def fit(self, seqs, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        s = self.s
        y = torch.tensor(y)
        perm = rng.permutation(len(y))
        dev, tr = perm[: len(y) // 10], perm[len(y) // 10:]

        self.net = RNNNet(self.vocab.init_matrix(s["emb"], self.seed), s["cell"], s["hidden"], s["bidir"],
                          s["readout"], s["word_drop"], freeze=s["emb"] == "frozen")
        opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad], lr=s["lr"])
        counts = torch.bincount(y[tr], minlength=20).float()
        loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (20 * counts))

        best, best_state, bad = -1.0, None, 0
        self.curve = []
        for epoch in range(self.max_epochs):
            self.net.train()
            order = tr[rng.permutation(len(tr))]
            total = 0.0
            for i in range(0, len(order), self.bs):
                b = order[i:i + self.bs]
                x, lengths = pad_batch([seqs[j] for j in b])
                opt.zero_grad()
                loss = loss_fn(self.net(x, lengths), y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            f1 = metrics(y[dev].numpy(), self.predict_proba([seqs[j] for j in dev]).argmax(1))["macro_f1"]
            self.curve.append({"epoch": epoch, "train_loss": total / len(tr), "dev_macro_f1": f1})
            if f1 > best:
                best, bad, self.best_epoch = f1, 0, epoch
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, seqs, bs=1024):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                x, lengths = pad_batch(seqs[i:i + bs])
                out.append(torch.softmax(self.net(x, lengths), 1).numpy())
        return np.concatenate(out)

# ------------------------------------------------------------------------------------------
# Phases
# ------------------------------------------------------------------------------------------

def setting_name(s):
    return (f"{s['cell']}-{s['emb']}-h{s['hidden']}{'-bi' if s['bidir'] else ''}-{s['readout']}"
            f"-wd{s['word_drop']}-lr{s['lr']:g}")


def sweep_settings():
    seen, out = set(), []
    for factor, values in SWEEP.items():
        for v in values:
            s = {**DEFAULT, factor: v}
            if setting_name(s) not in seen:
                seen.add(setting_name(s))
                out.append((factor, s))
    return out


def load_rows(path):
    return json.loads(path.read_text()) if path.exists() else []


def run_settings(rows, path, jobs, data, vocab, seqs):
    """Train each (tag, setting, seed, train subset) job unless it is already in `path` (so reruns resume)."""
    done = {(r["tag"], r["name"], r["seed"], r["frac"]) for r in rows}
    for tag, s, seed, frac in jobs:
        if (tag, setting_name(s), seed, frac) in done:
            continue
        idx = np.random.default_rng(seed).permutation(len(seqs["train"]))[: int(frac * len(seqs["train"]))]
        t0 = time.time()
        clf = RNNClassifier(vocab, s, seed).fit([seqs["train"][i] for i in idx], data["train"][1][idx])
        m = metrics(data["validation"][1], clf.predict_proba(seqs["validation"]).argmax(1))
        rows.append({"tag": tag, "name": setting_name(s), **s, "seed": seed, "frac": frac, **m,
                     "best_epoch": clf.best_epoch, "seconds": round(time.time() - t0, 1), "curve": clf.curve})
        path.write_text(json.dumps(rows, indent=1))
        print(f"{tag:<9} {setting_name(s):<40} frac={frac:<4} seed={seed}  val macro-F1={m['macro_f1']:.4f}"
              f"  acc={m['accuracy']:.4f}  epoch={clf.best_epoch}  {time.time() - t0:.0f}s", flush=True)
    return rows


def summary(rows, by):
    df = pd.DataFrame(rows)
    return df.groupby(by)[["macro_f1", "accuracy", "best_epoch"]].agg(["mean", "std"]).round(4)


def best_sweep_setting():
    """Highest mean validation macro-F1 in the step 4 sweep."""
    df = pd.DataFrame(load_rows(RESULTS / "step4_sweep.json"))
    name = df.groupby("name").macro_f1.mean().idxmax()
    r = df[df.name == name].iloc[0]
    return {k: (bool(r[k]) if k == "bidir" else r[k].item() if hasattr(r[k], "item") else r[k]) for k in DEFAULT}


def run_sweep(data, vocab, seqs, seeds=(0, 1, 2)):
    path = RESULTS / "step4_sweep.json"
    jobs = [(factor, s, seed, 1.0) for factor, s in sweep_settings() for seed in seeds]
    rows = run_settings(load_rows(path), path, jobs, data, vocab, seqs)
    print(summary(rows, ["tag", "name"]).to_string())
    print("best:", best_sweep_setting())


def run_datasize(data, vocab, seqs, emb, seeds=(0, 1, 2), fracs=(0.1, 0.3, 1.0)):
    from p2.step3_pool import PooledClassifier, pool_texts
    path = RESULTS / "step4_datasize.json"
    best = best_sweep_setting()
    jobs = [("datasize", {**best, "emb": mode}, seed, frac)
            for frac in fracs for mode in ("frozen", "trainable", "random") for seed in seeds]
    rows = load_rows(path)
    # full-data runs that the sweep already did (same setting, same seed) are reused, not retrained
    have = {(r["name"], r["seed"], r["frac"]) for r in rows if r["tag"] == "datasize"}
    for r in load_rows(RESULTS / "step4_sweep.json"):
        if (r["name"], r["seed"], 1.0) not in have and any(setting_name(s) == r["name"] and seed == r["seed"]
                                                          and frac == 1.0 for _, s, seed, frac in jobs):
            rows.append({**r, "tag": "datasize"})
            have.add((r["name"], r["seed"], 1.0))
    rows = run_settings(rows, path, jobs, data, vocab, seqs)

    # the step 3 model (frozen GloVe, mean+max pooling, MLP) on the same subsets, for reference
    done = {(r["seed"], r["frac"]) for r in rows if r["tag"] == "pool"}
    Xtr, Xva = pool_texts(data["train"][0], emb, "meanmax"), pool_texts(data["validation"][0], emb, "meanmax")
    for frac in fracs:
        for seed in seeds:
            if (seed, frac) in done:
                continue
            idx = np.random.default_rng(seed).permutation(len(Xtr))[: int(frac * len(Xtr))]
            clf = PooledClassifier("mlp", "adam_lr1e-4", True, seed).fit(Xtr[idx], data["train"][1][idx])
            m = metrics(data["validation"][1], clf.predict_proba(Xva).argmax(1))
            rows.append({"tag": "pool", "name": "embed-and-pool (step 3)", "emb": "frozen", "seed": seed,
                         "frac": frac, **m, "best_epoch": clf.best_epoch})
            print(f"pool frac={frac} seed={seed}  val macro-F1={m['macro_f1']:.4f}", flush=True)
    path.write_text(json.dumps(rows, indent=1))
    print(summary(rows, ["tag", "emb", "frac"]).to_string())


def prepare():
    data = load_data()
    emb = load_embeddings("glove200")
    vocab = Vocab(emb, [data[s][0] for s in data])
    seqs = {s: vocab.encode(data[s][0]) for s in data}
    return data, emb, vocab, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sweep", "datasize"])
    args = ap.parse_args()
    torch.set_num_threads(8)
    data, emb, vocab, seqs = prepare()
    print(f"vocabulary: {len(vocab.words)} rows (GloVe words found in our tweets + <pad> + <unk>)")
    if args.phase == "sweep":
        run_sweep(data, vocab, seqs)
    else:
        run_datasize(data, vocab, seqs, emb)


if __name__ == "__main__":
    main()
