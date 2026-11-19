# NLP1 Project 2: emoji prediction with pretrained word embeddings

Repository: https://github.com/pravallikalakkireddy/NLP-project-2

Same task as Project 1: read a tweet (TweetEval emoji) and guess which of **20 emojis** it had.
Same data handling too: the invisible `U+FE0F` leak is removed from all splits, the main metric is **macro-F1**,
and we use **exactly the same 5 folds × 3 seeds** on the 45k training tweets.
So every Project 2 score lines up with a Project 1 score for a paired comparison.

This README is also the **handover**: what was done in steps 1–3, the results, and what is left (steps 4–6).

| Step | Owner | Status |
|---|---|---|
| 1 Choose pretrained embeddings, check tokenisation and coverage | Pravallika | ✅ done |
| 2 Explore what the embeddings capture | Pravallika | ✅ done |
| 3 Embed-and-pool classifier, compare with best Project 1 model | Pravallika | ✅ done |
| 4 GRU/LSTM, frozen vs trainable embeddings, data size | teammate | ⏳ to do |
| 5 Hyperparameter search, seeds, variance, Bayesian ROPE | teammate | ⏳ to do |
| 6 Error analysis | teammate | ⏳ to do |
| Slides 1–3 | Pravallika | ✅ `slides/project2_slides_1-3.pdf` |
| Slides 4–6, merge into one PDF | together | ⏳ to do |

**Deadline:** 27 Nov, 23:59. Slides PDF (max 6 slides, backup slides for the oral) + link to a public repo.

---

## How to run

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
sh p2/download_embeddings.sh            # ~4.5 GB of word vectors into data/ (5-10 minutes)
.venv/bin/python -m p2.step1_coverage   # step 1, ~3 min the first time (builds small caches), then ~1 min
.venv/bin/python -m p2.step2_explore    # step 2, ~10 s
.venv/bin/python -m p2.step3_pool sweep # step 3a, ~2 min
.venv/bin/python -m p2.step3_pool cv    # step 3b, ~30 s
```

Everything is deterministic: a full clean rerun gives identical numbers.
`data/` (vectors) and `.venv/` are not in git (`.gitignore`).

## Files

```
p2/
  common.py              shared code: data + leak removal, Project 1's folds, GloVe tokeniser, embedding loader, metrics
  step1_coverage.py      step 1: vocabulary coverage per embedding × tokenisation, most common missing words
  step2_explore.py       step 2: neighbours, emoji vectors, class centroids, limitations
  step3_pool.py          step 3: embed-and-pool classifier (sweep on validation, then 5 folds × 3 seeds + test)
  download_embeddings.sh downloads the word vectors
results/
  step1_coverage.json    coverage numbers
  step2_explore.json     neighbours, similarities, centroid matrix
  step3_sweep.json       every sweep run (setting, seed, validation macro-F1 / accuracy)
  step3_cv/              final model: cv_results.json, test_summary.json, test predictions, per-tweet table
  p1_cv_results.json     Project 1's 12 systems × 15 CV scores (copied from the Project 1 repo, for step 5)
  figures/               the charts used on the slides
slides/
  project2_slides_1-3.pdf    slides 1-3
```

---

## Checklist: the course notebook's questions for steps 1–3

| Notebook asks | Answer | Where |
|---|---|---|
| 1. Find an embedding model that fits your data (language, kind of text) | GloVe Twitter: English tweets, like ours | Step 1, slide 1 |
| 1. Evidence your choice is a good one | Compared with GloVe Wikipedia, fastText, word2vec: best coverage, best classifier (+3.4 vs GloVe Wikipedia) | Step 1, slide 1 |
| 1. Can you tokenise coherently with the model's definition of a token? | Yes: Python version of Stanford's tweet preprocessing; 49% → 96% coverage | Step 1, slide 1 |
| 1. Does the vocabulary give enough coverage? | 93.8% of words, 95.7% of tokens; only glued hashtags/places missing | Step 1, slide 1 |
| 2. Good at similarities relevant to the task? | Tweet-sense neighbours (babe → bby, ily), ❤ next to `<heart>`, class centroids | Step 2, slide 2 |
| 2. Task-specific evidence it beats a reasonable alternative? | Nearest-centroid 12.5 vs 8.9–11.9; same classifier 20.7 vs 17.3–18.8 | Steps 1–2, slides 1–2 |
| 2. Limitations of the model? | Opposites similar, one vector per word, numbers merged, averaging loses negation | Step 2, slide 2 |
| 2. Limitations troubling for your task? | Same-context emojis look identical: 💜 / 💙 are the hardest classes; negation in 7% of tweets; "100" barely matters (0.5%) | Step 2, slide 2 |
| 3. Which design choices matter (embedding size, architecture, optimisation)? | Size and domain most; mean+max pooling; learning rate more than optimiser; MLP ≈ linear | Step 3, slide 3 |
| 3. Better than last week? Did pretrained embeddings add? | CV: no (19.9 vs 21.6). Test: yes (20.0 vs 19.0 single, 20.5 vs 19.5 ensemble). Formal test in step 5 | Step 3, slide 3 |

## Step 1: which embeddings, and does the tokenisation match?

**Choice: GloVe Twitter** (Stanford), trained on 2 billion tweets, 1.2M words, in 4 sizes (25/50/100/200 dimensions).
The 4 sizes also give step 3's embedding-size comparison for free.

**Compared with 3 alternatives**, each read with the tokenisation that matches how it was trained:

| Embedding | Trained on | Tokenisation we use |
|---|---|---|
| **GloVe Twitter 200d** | 2B tweets | GloVe's tweet tokeniser (below) |
| GloVe Wikipedia + Gigaword 200d | Wikipedia + news | lowercase, `don't` → `do n't`, no tweet tags |
| fastText wiki-news 300d | news + Wikipedia | cased, `don't` → `don 't` |
| word2vec Google News 300d | Google News | cased, `don't` → `don 't` |

GloVe Wikipedia is the cleanest comparison: **same method and same size as GloVe Twitter, only the training data differs**.

**Results** (45k training tweets; figure `results/figures/step1_embedding_comparison.png`):

| | GloVe Twitter | GloVe Wikipedia | fastText news | word2vec News |
|---|---|---|---|---|
| words with a vector (running words, punctuation left out) | **93.8%** | 91.2% | 91.3% | 85.6% |
| distinct words with a vector | **52.3%** | 42.8% | 50.2% | 49.4% |
| label emojis with a vector | ❤ ✨ ☀ (useful) | none | 17 of 20 (noise, see step 2) | none |
| same step 3 classifier, validation macro-F1 (3 seeds) | **20.7** | 17.3 | 18.8 | 18.2 |
| nearest-centroid classifier, validation macro-F1 (step 2) | **12.5** | 10.7 | 8.9 | 11.9 |

**Which words are missing** matters more than how many (most common missing words):
- GloVe Twitter: goldengatebridge, lalife, spaceneedle, lajolla. Only glued hashtags and place names.
- GloVe Wikipedia: selfie, lasvegas, dtla, repostapp, also bday, lmao. Everyday tweet words are missing.
- fastText: losangeles, lasvegas, repostapp, nofilter. Mostly glued hashtags.
- word2vec: to, a, and, of. Its authors dropped these common words, and it has no numbers either.

So: GloVe Twitter covers the most words, misses only unimportant ones, and gives the best classifier.
Same method, same size, only tweets instead of Wikipedia as training data: **+3.4 macro-F1 points**.

**Tokenisation has to match how the vectors were trained.** GloVe Twitter was trained on tweets run through
Stanford's preprocessing script, so `glove_tokenize()` in `common.py` does the same:
lowercase · `@user`→`<user>` · URL→`<url>` · `#tag`→`<hashtag> tag` · numbers→`<number>` ·
`soooo`→`so <elong>` · `!!!`→`! <repeat>` · ALLCAPS→`allcaps <allcaps>` · smileys→`<smile>` etc. ·
`don't`→`do n't`, `it's`→`it 's` (GloVe's vocabulary has `n't` and `'s`, not `don't`).
Project 1's BPE tokenizer is no use here: GloVe has vectors for whole words, not for BPE pieces.

Token coverage (share of all running tokens, punctuation included) for each tokenisation:

| Tokenisation | GloVe Twitter | GloVe Wikipedia | fastText news | word2vec News |
|---|---|---|---|---|
| split on spaces only | 49.1% | 48.2% | 73.7% | 68.2% |
| split off punctuation | 70.4% | 69.8% | **93.8%** | **75.2%** |
| + lowercase | 94.1% | 93.2% | 93.0% | 73.7% |
| + `don't` → `do n't` | 94.4% | **93.4%** | 92.7% | 74.0% |
| + GloVe tags (`<user>`, `<hashtag>`, …) = our GloVe tokeniser | **95.7%** | 79.6% | 78.8% | 62.7% |

- Matching the tokenisation roughly doubles GloVe Twitter's coverage (49% → 96%).
- Each embedding is best with its own tokenisation (bold), which is the one we use for it.
- Missing words are skipped when pooling. Every tweet still has at least one known word.

## Step 2: what do the embeddings capture?

Full output: `results/step2_explore.json`.

- **Neighbours follow the training domain.** For most task words the two models' top-10 neighbour lists share only 0–3 words.
  | word | GloVe Twitter | fastText news |
  |---|---|---|
  | babe | bby, babes, sweetie, ily | baby, girl, mama, mommy |
  | happy | birthday, day, bday, wish | unhappy, happier, glad, thankful |
  | beach | pool, resort, lake, sunset | beaches, beachfront, shore, shoreline |
  | lit | mon, dans, matin, ton (French!) | lighted, illuminated, brightly, lights |
- **Emoji vectors.** GloVe's ❤ is next to `<heart>`, ♥, love: a useful vector. fastText has vectors for 17 of our 20 emojis,
  but they are noise (their nearest neighbours are rare symbols like ◕ ็ у), because emojis are rare in news text.
  Having a vector is not the same as having a useful one.
- **Similarities that matter for the task: class centroids.** We averaged the GloVe vectors of all training tweets of each emoji
  (after removing the overall average) and compared the 20 averages (figure `results/figures/step2_centroid_similarity.png`).
  camera ≈ camera_flash (cosine 0.99), joy_tears ≈ wink (0.86), grin ≈ wink_tongue (0.84):
  in averaged-vector space these classes look the same, which predicts the confusions (Project 1 also confused 📸 → 📷 35% of the time).
  A nearest-centroid classifier (no training, just "which class average is closest?") gets
  **12.5 macro-F1 with GloVe Twitter**, against 11.9 (word2vec), 10.7 (GloVe Wikipedia) and 8.9 (fastText) on validation,
  so GloVe Twitter carries the most signal for this task.
- **Limitations, and which ones actually trouble our task.** We checked each one against our tweets
  (`how_often` in `step2_explore.json`) and the step 3 model's per-emoji test F1 (`per_class_f1_ensemble` in `test_summary.json`).
  - **Troubling:** emojis used in the same contexts get the same averaged vector (camera ≈ camera_flash 0.99), and the heart
    colours are near-synonyms (blue–purple 0.82). 💜 is the hardest class of all 20 (test F1 5.6) and 💙 is among the
    hardest (9.6). The words "blue" / "purple" themselves appear in only ~4% of those tweets, so the model has to rely on
    context, and the context is the same for every heart.
  - **A bit troubling:** averaging forgets word order and negation ("i am so happy today" vs "i am not happy today": cosine 0.986).
    7% of training tweets contain a negation (not / no / never / n't).
  - **General limitations, small for this task:** opposites are neighbours (love–hate 0.68, good–bad 0.80, happy–sad 0.58);
    one vector per word (GloVe's `lit` is the French word for "bed", not slang for "amazing");
    GloVe turns every number into `<number>`, so "100" loses its link to 💯, but only 0.5% of 💯 tweets contain "100".

## Step 3: embed and pool

`tweet → GloVe tokens → frozen word vectors → pool → standardise → classifier → 20 emojis`

The embeddings are **frozen** in step 3 (trainable embeddings are step 4), so the pooled vectors are computed once.
Training: cross-entropy, batch 128, up to 30 epochs; the best epoch is chosen on 10% of the training data by macro-F1
(like Project 1, the scored data is never used for that).

**3a. Tuning on the validation set** (`step3_pool.py sweep`, figure `results/figures/step3_sweep.png`).
Start from GloVe 200d · mean+max pooling · MLP (256 units, dropout 0.3) · Adam (lr 1e-3) · class-weighted loss,
and change **one factor at a time**, 3 seeds each. Validation macro-F1 (%):

| Factor | Results |
|---|---|
| embedding | GloVe Twitter 25d 14.6 · 50d 17.5 · 100d 19.3 · **200d 20.7** · GloVe Wikipedia 200d 17.3 · fastText 300d 18.8 · word2vec 300d 18.2 |
| pooling | mean 19.4 · max 17.5 · **mean+max 20.7** |
| classifier | linear 20.5 · **MLP 20.7** · 2-layer MLP 19.9 |
| optimiser + learning rate | **Adam lr 1e-4 21.0** · Adam lr 1e-3 20.7 · Adam lr 1e-2 16.5 · SGD lr 0.01 20.8 · SGD lr 0.05 20.9 · SGD lr 0.05 + momentum 0.9 18.6 |
| loss | **weighted 20.7** (accuracy 21%) · plain 20.8 (accuracy 31%) |

What it shows:
- Embedding size matters most, and more dimensions keep helping up to 200d.
- Domain beats size: all three non-Twitter embeddings, two of them 300d, are worse than GloVe Twitter 100d.
- Mean+max pooling is best: the mean gives the overall topic, the max keeps strong single-word signals.
- An MLP is barely better than a linear model, and a deeper MLP is worse.
- The learning rate matters more than the optimiser. Adam and SGD reach the same score with a suitable learning rate.
  Too large a step (Adam 1e-2, or SGD 0.05 with momentum, which acts like a larger step) overshoots: the best epoch is then
  one of the first few and the score drops by 2–4 points.
- The weighted loss does not change macro-F1 here, but it costs 10 points of accuracy (it predicts rare emojis more often).

**Chosen setting** (highest mean validation macro-F1, decided before looking at the test set):
**GloVe 200d · mean+max · MLP(256) · Adam (lr 1e-4) · weighted loss.**
Caveat for the oral: the top settings (Adam 1e-4, SGD 0.05, Adam 1e-3, plain loss) are within ~0.3 points of each other,
which is within seed noise, and "one factor at a time" does not test combinations (e.g. Adam 1e-4 + plain loss).
That is what step 5's search is for.

**3b. Comparison with the best Project 1 model** (`step3_pool.py cv`, figure `results/figures/step3_vs_project1.png`).
Same 5 folds × 3 seeds on the training tweets; each of the 15 fold models also predicts the 50k test tweets once.

| macro-F1 (%) | Project 1 best: logistic · BoW+bigrams | Project 2: GloVe 200d · mean+max · MLP |
|---|---|---|
| cross-validation (15 scores) | **21.6 ± 0.4** | 19.9 ± 0.4 |
| test, single fold models (mean of 15) | 19.0 | **20.0** |
| test, majority-vote ensemble of 15 | 19.5 | **20.5** |
| test accuracy, ensemble | 27.8% | 22.1% |

(Project 1 numbers come from the Project 1 repo, `project/analysis_teammate2.ipynb`.)

- In cross-validation, Project 1's bag-of-words model is better.
- On the test set, the embedding model is better. Bag-of-words loses 2.6 points from CV to test; the embedding model stays level (19.9 → 20.0).
- A likely reason: 6.8% of test tokens never occur in the training tweets, and about 3% of test tokens are such unseen words
  that still have a GloVe vector. Bag-of-words knows nothing about those words; the pretrained vectors do.
  This is a hypothesis; step 6 can check it.
- The embedding model's accuracy is lower because of the weighted loss (it trades accuracy for rare-class recall).
- Whether the CV difference is "real" is for the Bayesian comparison in step 5.

---

## What is saved for the teammate (steps 5 and 6)

All in `results/step3_cv/`:

| File | What it is |
|---|---|
| `cv_results.json` | 15 rows (seed 0–2 × fold 0–4): `macro_f1`, `accuracy`, `best_epoch`. Same format and same folds as Project 1. |
| `test_summary.json` | chosen setting, CV mean/sd, test scores (single models and ensemble) |
| `test_preds/s{seed}_f{fold}.npy` | each fold model's predicted label for the 50k test tweets (int8) |
| `test_ensemble_probs.npy` | average of the 15 models' probabilities, shape 50000 × 20 (float16) |
| `y_test.npy` | true test labels |
| `test_examples.csv` | one row per test tweet: `text`, `label`, `n_tokens`, `n_oov`, `oov_rate`, `n_chars`, `pred_ensemble`, `correct`, `p_label` (probability given to the true emoji) |

And `results/p1_cv_results.json`: Project 1's 15 CV scores for all 12 systems (pick `system == "logistic | bow+bigrams"`).

---

## What is left (teammate)

### Step 4: GRU / LSTM, frozen vs trainable embeddings, data size
- Reuse `common.py`: `load_data()`, `load_embeddings("glove200")`, `emb.tokenize(text)`, `emb.index` (word → row), `emb.vectors`.
- Turn each tweet into a list of row ids (skip OOV words, or map them to one extra all-zero row), pad to the batch's longest tweet.
- `nn.Embedding.from_pretrained(torch.tensor(emb.vectors), freeze=True)` for frozen, `freeze=False` for trainable, then `nn.GRU` / `nn.LSTM`
  (e.g. 1 layer, 128 hidden, take the last hidden state or max-pool the outputs) → `nn.Linear(…, 20)`.
- Data size: train on 10% / 25% / 50% / 100% of the training tweets. Expectation to test: frozen pretrained vectors help most when data is small;
  trainable ones may overfit on small data.
- Compare against the step 3 model (validation macro-F1, then CV like step 3).
- Use `common.kfold_ids(len(y), 5, seed)` with seeds 0, 1, 2 so the folds match Projects 1 and 2.

### Step 5: hyperparameter search, seeds, variance, ROPE
- Search on the validation set (e.g. learning rate, hidden size, dropout, pooling), with several seeds per setting; report the spread across seeds.
- Final candidates: 5 folds × 3 seeds, same as `step3_pool.py cv` (you can copy `run_cv`).
- Bayesian comparison: `BenavoliCorrelatedT(scores_A, scores_B, runs=3, names=[…])` from the course repo's `part1/analysis.py`,
  with the 15 scores sorted by (seed, fold) so they pair up. Project 1 used **ROPE = 0.01** (1 macro-F1 point).
  Comparisons to include: step 3 model vs Project 1 best; GRU/LSTM vs step 3; frozen vs trainable.

### Step 6: error analysis
- `test_examples.csv`: macro-F1 / accuracy by tweet length and OOV rate (bins), confusion matrix from `pred_ensemble`,
  per-emoji F1, and a few wrong tweets with high confidence (`p_label` low).
- Check the hypothesis from step 3: does the embedding model do better than Project 1 on tweets with words unseen in training?
  Project 1's test predictions are written by `project/run_experiments.py` in the Project 1 repo (not committed there; rerun
  `--encoders bow+bigrams --models logistic`).
- Our first look: accuracy hardly changes with OOV rate (21.5% for tweets with no OOV word, 21–23% with some).

### Slides and submission
- Slides 4–6: the editable PowerPoint of slides 1–3 (same design as Project 1) is not in the repo; ask Pravallika for it and add slides 4–6 there.
- Slide 1 already has the repo link: github.com/pravallikalakkireddy/NLP-project-2
- Export the 6 slides to one PDF (Keynote/PowerPoint: File → Export → PDF).

---

## Oral exam: likely questions on steps 1–3

| Question | Short answer |
|---|---|
| Why GloVe Twitter? | Compared with GloVe Wikipedia, fastText and word2vec: most words covered (94%), it only misses glued hashtags, and the same classifier scores best (20.7 vs 17.3–18.8). Against GloVe Wikipedia (same method and size) only the training data differs: +3.4 points. |
| Why not reuse the Project 1 tokenizer? | GloVe has vectors for words, not BPE pieces. The tokenisation must match how the vectors were trained. |
| What is token vs type coverage? | Token: share of all word occurrences that have a vector (96%). Type: share of distinct words (52%); the missing ones are rare. |
| What happens to OOV words? | They are skipped when pooling. |
| Why compare with those three? | Different training data (Wikipedia, news) and methods (fastText, word2vec). It shows that the domain matters more than the method or the dimension. |
| What do embeddings miss? | Opposites are similar (love/hate), one vector per word (lit), numbers merged, and averaging loses order and negation. |
| Why mean+max pooling? | The mean keeps the overall topic, the max keeps strong single-word signals; together best on validation. |
| Why frozen embeddings? | Step 3 tests pretrained knowledge only; trainable embeddings are step 4. It also makes training fast. |
| Why is CV worse but test better than Project 1? | Bag-of-words fits the training vocabulary well but cannot handle unseen words; pretrained vectors generalise. Hypothesis, checked in step 6. |
| Why a weighted loss? | Classes are imbalanced and our metric is macro-F1. Here it gave the same macro-F1 but lower accuracy. |


