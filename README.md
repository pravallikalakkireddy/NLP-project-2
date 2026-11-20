# NLP1 Project 2: emoji prediction with pretrained word embeddings

Repository: https://github.com/pravallikalakkireddy/NLP-project-2

Same task as Project 1: read a tweet (TweetEval emoji) and guess which of **20 emojis** it had.
Same data handling too: the invisible `U+FE0F` leak is removed from all splits, the main metric is **macro-F1**,
and we use **exactly the same 5 folds × 3 seeds** on the 45k training tweets.
So every Project 2 score lines up with a Project 1 score for a paired comparison.

This README documents all six steps: what we did, the results, and the answers to the course notebook's questions.

| Step | Owner | Status |
|---|---|---|
| 1 Choose pretrained embeddings, check tokenisation and coverage | Pravallika | ✅ done |
| 2 Explore what the embeddings capture | Pravallika | ✅ done |
| 3 Embed-and-pool classifier, compare with best Project 1 model | Pravallika | ✅ done |
| 4 GRU/LSTM, frozen vs trainable embeddings, data size | Ana | ✅ done |
| 5 Hyperparameter search, seeds, variance, Bayesian ROPE | Ana | ✅ done |
| 6 Error analysis | Ana | ✅ done |
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
sh p2/run_steps4-6.sh                   # steps 4-6, ~2 h on a laptop CPU (every phase resumes if stopped)
.venv/bin/python -m p2.make_figures     # slide figures for steps 4-6, ~10 s
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
  step4_rnn.py           step 4: GRU / LSTM encoder (sweep on validation, data-size experiment)
  step5_search.py        step 5: random search, 5 folds x 3 seeds for the final RNNs, Bayesian comparisons
  step6_errors.py        step 6: tweet features, scores per feature bin, Bayesian correctness model, examples
  make_figures.py        slide figures for steps 4-6
  run_steps4-6.sh        runs steps 4-6 in order
  download_embeddings.sh downloads the word vectors
part1/, part2/           the course's analysis.py files, unchanged (Bayesian t-test; Bayesian correctness model)
results/
  step1_coverage.json    coverage numbers
  step2_explore.json     neighbours, similarities, centroid matrix
  step3_sweep.json       every sweep run (setting, seed, validation macro-F1 / accuracy)
  step3_cv/              final model: cv_results.json, test_summary.json, test predictions, per-tweet table
  p1_cv_results.json     Project 1's 12 systems × 15 CV scores (copied from the Project 1 repo, for step 5)
  step4_sweep.json       step 4 sweep (every setting x seed: validation macro-F1, training curve)
  step4_datasize.json    step 4 data size (10/30/100% of the training tweets)
  step5_search.json      step 5 random search
  step5_cv/<setting>/    final RNNs: cv_results.json, test_summary.json, ensemble test probabilities, per-tweet table
  step5_compare.json     Bayesian comparisons (ROPE 0.5 / 1 / 2 points) and variance breakdown
  step6_*.json           error analysis: binned scores, Bayesian model summary, per-emoji F1, examples
  step6_features.csv.gz  the 10 analysis features + every system's prediction, one row per test tweet
  p1_test_preds/         Project 1's BoW model test predictions (15 fold models), for step 6
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

## Checklist: the course notebook's questions for steps 4–6

| Notebook asks | Answer | Where |
|---|---|---|
| 4. Which design choices matter? | Pretrained vectors (+4 over random), readout (mean+max over all states), GRU over LSTM; bidirectional and word dropout only help together | Step 4, slide 4 |
| 4. Data size, embeddings trained or not, fixed or trainable? | 10/30/100% of the data × frozen / trainable / random: pretraining helps at every size, most at 10%; frozen ≥ trainable | Step 4, slide 4 |
| 4. Did you struggle with over-fitting? | Trainable embeddings (8M extra parameters) peak earlier (epoch 3 vs 6) and score 1.2 lower; early stopping on 10% of the training data | Step 4 |
| 4. How far can it go? Better than embed-and-pool? | Yes: +2.5 macro-F1 in CV (P = 0.999), test 23.9 vs 20.5 | Steps 4–5, slides 4–5 |
| 5. How did you structure the hyperparameter search? | Coarse to fine: one factor at a time → random combinations of the competitive values → 5 folds × 3 seeds for the winner | Step 5, slide 5 |
| 5. Are you keeping an eye on experimental variance? | 3 seeds for every setting (seed noise up to ±1 point), winner's curse checked by re-estimating, Bayesian ROPE on paired CV scores, seed vs fold variance | Step 5, slide 5 |
| 6. Which analysis features? Which did you test, how did it go? | 10 features (length, unseen words, rare words, `@ Place`, hashtags, negation, sentiment, …), binned scores + the course's Bayesian correctness model | Step 6, slide 6 |
| 6. What would you prioritise next? | Treat 📷/📸 and the heart colours as near-duplicates, contextual embeddings, split glued hashtags | Step 6, slide 6 |

## Step 4: recurrent encoder (GRU / LSTM)

`tweet → GloVe tokens → word vectors → GRU or LSTM → readout → MLP → 20 emojis`

Only the encoder changes compared with step 3: same tokeniser, same GloVe 200d, same MLP head (256, dropout 0.3),
same class-weighted loss, same "best epoch on 10% of the training data". Plus gradient clipping (standard for RNNs).
Vocabulary: every GloVe word that occurs in our tweets (41,832) + `<pad>` + `<unk>` (zero vectors; 4% of tokens).

- **Embeddings:** *frozen* (GloVe, not updated), *trainable* (GloVe, fine-tuned) or *random* (same size, no pretraining).
- **Readout** (how the per-word states become one tweet vector): `last` state, `max`, or `meanmax` over all states.
- **Word dropout:** blank out random whole words while training (the course's `LSTMEncoder` regulariser).

**4a. Sweep** (`step4_rnn.py sweep`, figure `results/figures/slide4_sweep.png`): change one factor of the default
(GRU · frozen · 128 units · last state · no word dropout · Adam 1e-3), 3 seeds each, validation macro-F1 (%):

| Factor | Results |
|---|---|
| cell | **GRU 22.8** · LSTM 21.5 |
| embeddings | **frozen 22.8** · trainable 21.2 · random 18.9 |
| direction | **one way 22.8** · both ways 22.7 |
| readout | last 22.8 · max 22.7 · **mean+max 23.2** |
| word dropout | **0: 22.8** · 0.25: 21.9 |

Hidden size (128) and learning rate (Adam 1e-3) were kept fixed to save compute; step 3 already studied the learning rate.

**4b. Data size** (`step4_rnn.py datasize`, figure `results/figures/slide4_datasize.png`), validation macro-F1 (%):

| training tweets | GRU frozen | GRU trainable | GRU random | embed-and-pool |
|---|---|---|---|---|
| 10% (4.5k) | **17.0** | 16.7 | 13.0 | 15.4 |
| 30% (13.5k) | **21.1** | 19.5 | 15.8 | 19.2 |
| 100% (45k) | **23.2** | 22.2 | 19.9 | 21.0 |

What it shows:
- **Pretraining matters most.** Random vectors are 3–4 points worse at every size; with 10% of the data the frozen GRU is
  still 4 points ahead of them.
- **Frozen ≥ trainable.** Fine-tuning adds 8M parameters and over-fits: its best epoch comes earlier (3 vs 6 on full data)
  and it scores 1–1.6 points lower. 45k tweets is not enough to improve on vectors learned from 2B tweets.
- **The GRU beats embed-and-pool at every data size**, so word order helps even with little data.
- **Readout:** mean+max over all states is best and very stable (sd 0.05); the last state alone forgets early words.
- **LSTM < GRU** at the same settings (we did not tune the LSTM separately).

## Step 5: hyperparameter search, variance, Bayesian comparison

**How the search is structured** (coarse to fine):
1. **One factor at a time** (step 3 and step 4 sweeps): main effects, 3 seeds each.
2. **Random search over combinations** (`step5_search.py search`): for every factor, keep the values within 1 point of its
   best (smaller gaps are seed noise), then 6 random combinations × 3 seeds. Space: GRU · frozen · {one way, both ways} ·
   {last, max, mean+max} · {word dropout 0, 0.25}. Figure `results/figures/slide5_search.png`.
3. **Re-estimate the winner** with Project 1's exact 5 folds × 3 seeds (`step5_search.py cv`), and its trainable twin.

Best on validation: **bidirectional GRU · frozen · mean+max · word dropout 0.25: 23.5** (step 4's best: 23.2).
Both ways and word dropout did not help on their own (step 4), only in combination; that is what the combination search is for.

**Variance.** Every setting has 3 seeds; the seed alone moves a setting by up to ±1 point (default GRU: 21.9 / 23.6 / 22.8),
so validation gaps under ~1 point are not evidence. Picking the best of 14 noisy settings is optimistic (**winner's curse**):
23.5 on validation becomes 22.4 in fresh cross-validation. In CV the spread comes from both folds and seeds, and how much the
seed adds differs per system (sd over seeds 0.0–0.4 points; `variance` in `step5_compare.json`), so one seed is never enough.

**Bayesian comparison** (`step5_search.py compare`): correlated t-test on the 15 paired CV scores (same folds and seeds for
every system), `BenavoliCorrelatedT(..., runs=3)`, **ROPE ±1 point** as in Project 1 (0.5 and 2 points in `step5_compare.json`).

| A vs B | B − A | P(A better) | P(equal) | P(B better) |
|---|---|---|---|---|
| embed-and-pool vs **BiGRU** | +2.5 | 0.000 | 0.001 | **0.999** |
| Project 1 BoW vs BiGRU | +0.8 | 0.001 | **0.64** | 0.36 |
| BiGRU frozen vs trainable | −1.2 | **0.65** | 0.35 | 0.000 |
| Project 1 BoW vs embed-and-pool | −1.7 | **0.96** | 0.04 | 0.000 |
| step 4 GRU vs search BiGRU | +0.4 | 0.000 | **0.98** | 0.02 |

- The recurrent encoder is clearly better than embed-and-pool.
- Against Project 1, CV says "practically equal or better, not worse"; on the test set the BiGRU is clearly ahead (below).
- Frozen embeddings are at least as good as trainable ones.
- The random search's gain over step 4 is **practically zero** (P(equal) = 0.98): both GRUs are equally good. We report the
  search winner because "highest mean validation score" was the rule fixed before running CV.

**Test set** (used once, after all choices; figure `results/figures/slide5_cv_test.png`):

| macro-F1 (%) | Project 1 BoW | embed-and-pool | BiGRU |
|---|---|---|---|
| cross-validation (15 scores) | 21.6 ± 0.4 | 19.9 ± 0.4 | **22.4 ± 0.6** |
| test, single fold models (mean of 15) | 19.0 | 20.0 | **22.4** |
| test, majority vote of 15 | 19.5 | 20.5 | **23.9** |
| test accuracy, majority vote | **27.8%** | 22.1% | 24.6% |

Final model: **bidirectional GRU (128 per direction) on frozen GloVe Twitter 200d, mean+max readout, word dropout 0.25**,
`results/step5_cv/gru-frozen-h128-bi-meanmax-wd0.25-lr0.001/`.

## Step 6: error analysis

Three systems on the same 50k test tweets (`step6_errors.py`): Project 1 BoW (majority vote), embed-and-pool, BiGRU
(both ensembles of their 15 CV models). Features, each with the reason we expected it to matter:

| Feature | Why |
|---|---|
| `n_tokens` | longer tweets: more context, but harder to squeeze into one vector |
| `unseen_rate` | words never seen in training: BoW knows nothing about them, GloVe does (step 3's hypothesis) |
| `oov_rate` | tokens without a GloVe vector (zero vectors for the embedding models) |
| `log_freq` | how common the tweet's words are in training |
| `location` | Instagram-style "… @ Place" ending (49% of tweets) |
| `n_hashtags` | glued words (`#sundayfunday`), often unknown to GloVe |
| `negation` | averaging ignores it; a GRU can in principle use it |
| `sentiment` | positive − negative opinion words (Hu & Liu lexicon, as in the tutorial) |
| `intensity` | elongation, `!!!`, ALL CAPS |
| `user_mention` | tweets addressed to someone (😘 😉) |

**Two tests:** (a) macro-F1 per feature bin (`step6_bins.json`, figure `slide6_bins.png`); (b) the course's Bayesian
correctness model (`part2/analysis.py`): "is the prediction correct?" ~ standardised features, one weight vector per system,
on 10k test tweets (2 chains, r̂ ≤ 1.002, 0 divergences; `step6_bda.json`, figure `slide6_bda.png`).

Findings:
- **The BiGRU is better on every kind of tweet**: +3–4 points over the other two in every bin of every feature.
- **Unseen words (step 3's hypothesis): confirmed.** From tweets without unseen words to tweets with >10%:
  BoW −2.6 points, embed-and-pool −0.9, BiGRU −1.2. Pretrained vectors protect against vocabulary not seen in training.
- **Length** hurts BoW (weight −0.13) but not the BiGRU (+0.04); P = 1.0 that the two weights differ.
  Very short tweets (≤ 8 tokens) are the hardest for every system.
- **BoW relies on frequent words** (weight +0.19 for word frequency); the embedding models do not.
- **Negation** costs BoW 2.5 points and the BiGRU 1.3: a hint that word order helps, but weak (P = 0.70).
- **Per emoji** (`per_class_f1` in `step6_examples.json`): the BiGRU gains most on 🇺🇸 (+13 over embed-and-pool), 📷 (+9),
  🔥 (+8). It is weakest on ❤ (7 vs BoW's 40): the weighted loss stops it from predicting the most common emoji all the time,
  which is also why its accuracy is lower than BoW's while its macro-F1 is higher.
- **Biggest confusion:** 📸 predicted as 📷 47% of the time (centroid cosine 0.99 in step 2), then 😜 → 😉 (19%).

**What we would prioritise next:** treat 📷/📸 and the heart colours as near-duplicates (merge them, or report top-2);
contextual embeddings instead of one vector per word; split glued hashtags; tune the LSTM separately.

## Limitations and decisions to know about

- Compute: the step 4 sweep keeps hidden size and learning rate fixed, the search has 6 combinations, and the final CV
  has 2 systems (random embeddings are covered by the data-size runs). Seeds stay at 3 everywhere.
- The Bayesian correctness model scores accuracy-style correctness, so frequent emojis weigh more there than in macro-F1.
- Sweeps use the official validation set; cross-validation uses folds of the training set; the test set is used once.

---

## Slides and submission
- Slides 1–3: `slides/project2_slides_1-3.pdf`. Slides 4–6 use the figures `results/figures/slide4_*.png`,
  `slide5_*.png`, `slide6_*.png` (`slide5_search.png` and `slide6_bda.png` also work as backup slides).
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

## Oral exam: likely questions on steps 4–6

| Question | Short answer |
|---|---|
| What is the final model? | Bidirectional GRU (128 per direction) on frozen GloVe Twitter 200d, mean+max over the states, word dropout 0.25, same MLP head as step 3. |
| Why a GRU and not an LSTM? | Both tested at the same settings: GRU 22.8 vs LSTM 21.5. Fewer parameters, trains faster. We did not tune the LSTM separately. |
| What is the readout and why mean+max? | How the per-word states become one tweet vector. The last state forgets early words; mean+max over all states uses every position and was best and most stable. |
| Why frozen embeddings? | Trainable ones over-fit (best epoch 3 vs 6) and score 1.2 lower in CV (P = 0.65 frozen better, otherwise equal). 45k tweets is too little to improve on vectors from 2B tweets. |
| Why random embeddings at all? | A control: it measures what pretraining contributes (3–4 points, at every data size). |
| How did you structure the search? | Coarse to fine: one factor at a time → random combinations of the values within 1 point of the best → 5 folds × 3 seeds for the winner. 3 seeds per setting throughout. |
| Did the random search help? | It found the best validation score (23.5 vs 23.2), but in CV the two GRUs are practically equal (P = 0.98). It did show that both ways + word dropout only help together. |
| What is the winner's curse? | Picking the best of many noisy scores overestimates it: 23.5 on validation → 22.4 in CV. Fix: re-estimate on fresh folds and seeds. |
| Why the correlated t-test? | CV folds share training data, so their scores are correlated; a plain t-test would be over-confident. `runs=3` because of the 3 seeds. |
| Why ROPE = 1 point? | Smaller differences are within seed noise and would not change which system we use. Results at 0.5 and 2 points are in `step5_compare.json`. |
| Is the BiGRU better than Project 1? | CV: +0.8, practically equal (0.64) or better (0.36), not worse. Test: 23.9 vs 19.5. BoW fits its training vocabulary but generalises worse to unseen words. |
| Why is its accuracy lower than Project 1's? | Weighted loss: it predicts rare emojis more often and gives up ❤ (F1 7 vs 40). Macro-F1 is our metric, so that trade is intended. |
| What does a weight in the Bayesian correctness model mean? | The effect of a standardised feature on the log-odds that the system is right. Negative: the system struggles as the feature grows. Comparing weights across systems shows where they differ. |
| Why are 📷 and 📸 confused? | They appear in the same contexts (centroid cosine 0.99), so from text alone they are close to indistinguishable. |

