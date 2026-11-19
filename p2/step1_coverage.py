"""
Step 1: which pretrained embeddings fit our tweets, and does our tokenisation match theirs?

For every candidate embedding we measure vocabulary coverage on the training tweets
(type coverage = share of distinct tokens that have a vector, token coverage = share of running tokens),
under different tokenisations. Writes results/step1_coverage.json and prints a table.

Run:  .venv/bin/python -m p2.step1_coverage
"""
import json

from p2.common import (EMBEDDINGS, EMOJIS, RESULTS, coverage, glove_tokenize,
                       load_data, load_embeddings, ptb_tokenize, simple_tokenize, whitespace_tokenize)

TOKENISATIONS = {
    "whitespace": whitespace_tokenize,                                   # tweet.split(), no changes
    "simple": simple_tokenize,                                           # split off punctuation
    "simple_lower": lambda t: simple_tokenize(t, lower=True),            # + lowercase
    "ptb_lower": ptb_tokenize,                                           # + "don't" -> do n't
    "glove_twitter": glove_tokenize,                                     # + GloVe's special tokens
}


def main():
    data = load_data()
    train = data["train"][0]
    toks = {name: [f(t) for t in train] for name, f in TOKENISATIONS.items()}

    results = {}
    for emb_name in EMBEDDINGS:
        emb = load_embeddings(emb_name)
        results[emb_name] = {tok: coverage(toks[tok], emb) for tok in TOKENISATIONS}
        # with the tokenisation that matches this embedding (common.tokenizer_for), all tokens and words only
        matched = [emb.tokenize(t) for t in train]
        results[emb_name]["matched"] = coverage(matched, emb)
        results[emb_name]["matched_words_only"] = coverage([[w for w in ts if w[0].isalnum()] for ts in matched], emb)
        results[emb_name]["emojis_with_vector"] = [e for e in EMOJIS if e in emb]

    example = train[7]
    results["example"] = {"tweet": example, **{k: f(example) for k, f in TOKENISATIONS.items()}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "step1_coverage.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))

    print("token coverage per tokenisation")
    print(f"{'embedding':<13}" + "".join(f"{t:>15}" for t in TOKENISATIONS))
    for emb_name in EMBEDDINGS:
        print(f"{emb_name:<13}" + "".join(f"{results[emb_name][t]['token_coverage']:>15.1%}" for t in TOKENISATIONS))
    print("\nwith each embedding's matched tokenisation")
    for emb_name in EMBEDDINGS:
        r, w = results[emb_name]["matched"], results[emb_name]["matched_words_only"]
        print(f"{emb_name:<13} tokens {r['token_coverage']:.1%}  words only {w['token_coverage']:.1%}  "
              f"distinct words {w['type_coverage']:.1%}  emojis {''.join(results[emb_name]['emojis_with_vector'])}")
        print(f"{'':<13} most common missing words: {', '.join(w['top_oov'][:10])}")


if __name__ == "__main__":
    main()
