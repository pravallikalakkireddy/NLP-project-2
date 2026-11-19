#!/bin/sh
# Downloads GloVe Twitter (25/50/100/200d), fastText wiki-news 300d, GloVe Wikipedia 200d and
# word2vec Google News 300d (~4.5 GB) into data/embeddings/.
# Re-running resumes unfinished downloads.  Run from the project folder:  sh p2/download_embeddings.sh
mkdir -p data/embeddings
for n in glove-twitter-25 glove-twitter-50 glove-twitter-100 glove-twitter-200 fasttext-wiki-news-subwords-300 \
         glove-wiki-gigaword-200 word2vec-google-news-300; do
  curl -L -C - --retry 3 -o "data/embeddings/$n.gz" \
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/$n/$n.gz"
done
