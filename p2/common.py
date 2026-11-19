"""
Shared helpers for Project 2 (steps 1-3): data, tokenisation, embeddings, folds, metrics.

The data handling is identical to Project 1 (same dataset, same leak removal, same 5 folds x 3 seeds),
so Project 2 scores can be compared pair-by-pair with the Project 1 scores.
"""
import gzip
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
RESULTS = ROOT / "results"

EMOJIS = ["❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘",
          "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁", "🎄", "📸", "😜"]   # label 0..19
NAMES = ["red_heart", "heart_eyes", "joy_tears", "two_hearts", "fire", "smiling", "sunglasses", "sparkles",
         "blue_heart", "kiss", "camera", "US_flag", "sun", "purple_heart", "wink", "hundred", "grin",
         "xmas_tree", "camera_flash", "wink_tongue"]                         # plot labels (fonts lack emoji)

# ------------------------------------------------------------------------------------------
# Data (same as Project 1: TweetEval emoji, U+FE0F label leak removed from every split)
# ------------------------------------------------------------------------------------------

def clean_leak(text):
    return " ".join(text.replace("️", " ").split())


def load_data():
    """Returns dict split -> (list of texts, np.array of labels) for train / validation / test."""
    from datasets import load_dataset
    d = load_dataset("cardiffnlp/tweet_eval", "emoji")
    return {s: ([clean_leak(t) for t in d[s]["text"]], np.array(d[s]["label"]))
            for s in ("train", "validation", "test")}


def kfold_ids(n, k, seed):
    """Exactly Project 1's folds: shuffle 0..n-1 with `seed`, cut into k parts."""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def metrics(y_true, y_pred):
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred))}

# ------------------------------------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------------------------------------
# GloVe Twitter was trained on tweets preprocessed by Stanford's script (preprocess-twitter.rb):
# lowercase, and special tokens for users, urls, hashtags, numbers, smileys, ALLCAPS, elongations.
# To look words up in GloVe we have to tokenise our tweets the same way. glove_tokenize() is a
# short Python version of that script.

def _sub(pattern, repl, text, flags=0):
    return re.sub(pattern, repl, text, flags=flags)


def glove_tokenize(text):
    eyes, nose = r"[8:=;]", r"['`\-]?"
    t = text
    t = _sub(r"https?://\S+|www\.\S+", " <url> ", t)
    t = _sub(r"@\w+", " <user> ", t)
    t = _sub(rf"{eyes}{nose}[)d]+|[)d]+{nose}{eyes}", " <smile> ", t, re.I)
    t = _sub(rf"{eyes}{nose}p+", " <lolface> ", t, re.I)
    t = _sub(rf"{eyes}{nose}\(+|\)+{nose}{eyes}", " <sadface> ", t)
    t = _sub(rf"{eyes}{nose}[/|l*]", " <neutralface> ", t)
    t = _sub(r"<3", " <heart> ", t)
    t = _sub(r"[-+]?[.\d]*\d+[:,.\d]*", " <number> ", t)
    t = _sub(r"#(\w+)", r" <hashtag> \1 ", t)
    t = _sub(r"([!?.]){2,}", r" \1 <repeat> ", t)
    t = _sub(r"\b(\S*?)(\w)\2{2,}\b", r"\1\2 <elong> ", t)          # soooo -> so <elong>
    t = _sub(r"\b([A-Z]{2,})\b", lambda m: m.group(1) + " <allcaps> ", t)
    t = t.lower()
    # split clitics the way GloVe's tokeniser did: "don't" -> do n't, "it's" -> it 's
    return re.findall(r"<\w+>|\w+(?=n't\b)|n't\b|'\w+|\w+|[^\w\s]", t)


def simple_tokenize(text, lower=False):
    """Word/punctuation split, no special tokens (used for fastText): "don't" -> don 't"""
    return re.findall(r"'\w+|\w+|[^\w\s]", text.lower() if lower else text)


def ptb_tokenize(text, lower=True):
    """Word/punctuation split with GloVe's clitics ("don't" -> do n't), no tweet tags (used for GloVe Wikipedia)"""
    return re.findall(r"\w+(?=n't\b)|n't\b|'\w+|\w+|[^\w\s]", text.lower() if lower else text)


def whitespace_tokenize(text):
    return text.split()

# ------------------------------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------------------------------
# Files come from the gensim-data releases: word2vec format, a header line "<num words> <dim>", then one word
# + vector per entry (as text, or as raw float32 bytes for word2vec Google News).
# We only keep (a) the most frequent `top_k` words, for nearest-neighbour search, and (b) every word
# that occurs in our tweets. The result is cached as .npz so later loads take a second.

EMBEDDINGS = {
    "glove25": "glove-twitter-25", "glove50": "glove-twitter-50",
    "glove100": "glove-twitter-100", "glove200": "glove-twitter-200",
    "fasttext300": "fasttext-wiki-news-subwords-300",
    "glovewiki200": "glove-wiki-gigaword-200",              # step 1 comparison only
    "word2vec300": "word2vec-google-news-300",              # step 1 comparison only
}


class Embeddings:
    def __init__(self, name, words, vectors):
        self.name = name
        self.tokenize = tokenizer_for(name)          # the tokenisation that matches this embedding
        self.words = [str(w) for w in words]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.vectors = vectors.astype(np.float32)
        self.dim = vectors.shape[1]
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.unit = self.vectors / np.maximum(norms, 1e-8)

    def __contains__(self, w):
        return w in self.index

    def __getitem__(self, w):
        return self.vectors[self.index[w]]

    def similarity(self, a, b):
        return float(self.unit[self.index[a]] @ self.unit[self.index[b]])

    def neighbours(self, word_or_vec, k=10, search_top=50_000):
        """k most cosine-similar words among the `search_top` most frequent words."""
        v = self.unit[self.index[word_or_vec]] if isinstance(word_or_vec, str) else word_or_vec
        sims = self.unit[:search_top] @ (v / np.linalg.norm(v))
        out = []
        for i in np.argsort(-sims):
            if self.words[i] != word_or_vec:
                out.append((self.words[i], float(sims[i])))
            if len(out) == k:
                return out


def load_embeddings(name, top_k=100_000):
    cache = EMB_DIR / f"{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return Embeddings(name, z["words"], z["vectors"])
    data = load_data()
    keep = corpus_vocab([data[s][0] for s in data])
    words, vecs = [], []
    for i, (w, v) in enumerate(_read_vectors(EMB_DIR / f"{EMBEDDINGS[name]}.gz")):
        if i < top_k or w in keep:
            words.append(w)
            vecs.append(v())
    vecs = np.stack(vecs)
    np.savez(cache, words=np.array(words), vectors=vecs)
    return Embeddings(name, words, vecs)


def _read_vectors(path):
    """Yields (word, function returning its vector), so vectors we don't keep are never parsed."""
    with gzip.open(path, "rb") as f:
        n, dim = map(int, f.readline().split())
        binary = path.name.startswith("word2vec")
        for _ in range(n):
            if binary:                                   # word, space, dim float32 numbers
                word = b""
                while (c := f.read(1)) != b" ":
                    word += c
                raw = f.read(4 * dim)
                yield word.decode("utf-8", "replace").strip(), lambda raw=raw: np.frombuffer(raw, np.float32)
            else:                                        # "word v1 v2 ..."
                w, rest = f.readline().decode("utf-8", "replace").rstrip("\n").split(" ", 1)
                yield w, lambda rest=rest: np.array(rest.split(), dtype=np.float32)


def tokenizer_for(emb_name):
    """The tokenisation that matches how each embedding was trained (chosen in step 1)."""
    if emb_name.startswith(("fasttext", "word2vec")):
        return simple_tokenize                           # cased, these vocabularies keep capitals
    if emb_name.startswith("glovewiki"):
        return ptb_tokenize                              # lowercased, no tweet tags
    return glove_tokenize                                # GloVe Twitter


def corpus_vocab(texts_by_split):
    """All token types (under every tokenisation we try) in any split; used as `keep` when loading."""
    vocab = set()
    for texts in texts_by_split:
        for t in texts:
            vocab.update(glove_tokenize(t))
            vocab.update(simple_tokenize(t))
            vocab.update(simple_tokenize(t, lower=True))
            vocab.update(ptb_tokenize(t))
            vocab.update(ptb_tokenize(t, lower=False))
            vocab.update(whitespace_tokenize(t))
    return vocab | set(EMOJIS)


def oov_rate(tokens, emb):
    return sum(w not in emb for w in tokens) / max(len(tokens), 1)


def coverage(token_lists, emb):
    """Type coverage (share of distinct tokens found) and token coverage (share of running tokens)."""
    counts = Counter(w for toks in token_lists for w in toks)
    found = {w for w in counts if w in emb}
    return {"types": len(counts),
            "type_coverage": len(found) / len(counts),
            "token_coverage": sum(counts[w] for w in found) / sum(counts.values()),
            "top_oov": [w for w, _ in counts.most_common() if w not in emb][:25]}
