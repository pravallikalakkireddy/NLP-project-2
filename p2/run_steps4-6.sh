#!/bin/sh
# Runs steps 4-6 end to end (about 4-5 hours on a laptop CPU). Every phase resumes where it stopped.
# Run from the project folder:  sh p2/run_steps4-6.sh
set -e
PY=.venv/bin/python
mkdir -p logs
$PY -m p2.step4_rnn sweep      2>&1 | tee logs/step4_sweep.log
$PY -m p2.step4_rnn datasize   2>&1 | tee logs/step4_datasize.log
$PY -m p2.step5_search search  2>&1 | tee logs/step5_search.log
$PY -m p2.step5_search cv      2>&1 | tee logs/step5_cv.log
$PY -m p2.step5_search compare 2>&1 | tee logs/step5_compare.log
$PY -m p2.step6_errors         2>&1 | tee logs/step6_errors.log
