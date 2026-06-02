#!/usr/bin/env bash
# Four-way tqivf benchmark suite (tqivf vs tqflat vs ivfflat vs hnsw) on the-fire.
# Runs SIFT1M (L2), GloVe-200 (cosine), OpenAI-1M (cosine) SEQUENTIALLY so that
# concurrent index builds / loads don't contaminate the latency measurements.
# Each dataset writes its own CSV + log so partial results survive a failure.
set -u
cd ~/pgvector || exit 1

PY=python3
B=bench/tqflat_bench.py
RES=bench/results
mkdir -p "$RES"

# tqivf wide sweep + tuned hnsw + 16GB build memory, shared across datasets.
DB="--user max --host /var/run/postgresql --dbname tqtest"
SWEEP="--tqivf-probes-list 10 30 99 200 400 --tqivf-reranks 100 200 2000"
TQFLAT="--tqflat-bits 4 --tqflat-reranks 200 2000 --tqflat-tq-prod off"
HNSW="--hnsw-m 32 --hnsw-ef-construction 128 --hnsw-ef-search 100"
MEM="--maintenance-work-mem 16GB"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "================ SIFT1M (L2) $(ts) ================"
$PY $B \
  --fvecs-vectors ~/tqdata/sift/sift/sift_base.fvecs \
  --fvecs-queries ~/tqdata/sift/sift/sift_query.fvecs \
  --ivecs-gt      ~/tqdata/sift/sift/sift_groundtruth.ivecs \
  --metric l2 --no-exact --max-queries 500 \
  --methods tqflat,tqivf,hnsw,ivfflat \
  $DB $SWEEP $TQFLAT $HNSW $MEM \
  --csv $RES/sift1m_tqivf_x86.csv 2>&1 | tee $RES/sift1m_tqivf_x86.log
echo "SIFT1M exit=${PIPESTATUS[0]} $(ts)"

echo "================ GloVe-200 (cosine) $(ts) ================"
$PY $B \
  --hdf5 ~/tqdata/glove/glove-200-angular.hdf5 \
  --metric cosine --no-exact --max-queries 500 \
  --methods tqflat,tqivf,hnsw,ivfflat \
  $DB $SWEEP $TQFLAT $HNSW $MEM \
  --csv $RES/glove200_tqivf_x86.csv 2>&1 | tee $RES/glove200_tqivf_x86.log
echo "GloVe exit=${PIPESTATUS[0]} $(ts)"

echo "================ OpenAI-1M (cosine) $(ts) ================"
# 200 queries shipped; compute exact GT (no file GT) and report exact baseline too.
$PY $B \
  --dataset-vectors ~/tqdata/openai1m/base.npy \
  --dataset-queries ~/tqdata/openai1m/queries.npy \
  --metric cosine \
  --methods exact,tqflat,tqivf,hnsw,ivfflat \
  $DB $SWEEP $TQFLAT $HNSW $MEM \
  --csv $RES/openai1m_tqivf_x86.csv 2>&1 | tee $RES/openai1m_tqivf_x86.log
echo "OpenAI1M exit=${PIPESTATUS[0]} $(ts)"

echo "================ SUITE DONE $(ts) ================"
