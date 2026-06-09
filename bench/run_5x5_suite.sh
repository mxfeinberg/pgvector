#!/usr/bin/env bash
# 5x5 ANN benchmark suite: {tqflat, tqivf, tqhnsw, ivfflat, hnsw}
#                        x {agtalk, sift1m, gist1m, glove200, openai1m}
# Run on the-fire. Datasets run SEQUENTIALLY (one index in memory at a time) so
# concurrent builds don't contaminate latency or blow past RAM. Each dataset
# writes its own CSV + log, so partial results survive a failure.
#
#   bash bench/run_5x5_suite.sh                 # all 5 datasets
#   bash bench/run_5x5_suite.sh agtalk          # just one (or several) datasets
#
set -u
cd ~/pgvector || exit 1

PY=python3
B=bench/tqflat_bench.py
RES=bench/results
TQ=~/tqdata
mkdir -p "$RES"

DB="--user max --host /var/run/postgresql --dbname tqtest"
METHODS="tqflat,tqivf,tqhnsw,hnsw,ivfflat"
MAXQ="--max-queries 1000"
K="--k 10"

# Shared sweeps (Pareto curves per method).
# blocked (8-bit SIMD) tqflat layout requires tq_prod=off (QJL unsupported there)
TQFLAT="--tqflat-bits 4 --tqflat-reranks 200 2000 --tqflat-tq-prod off"
TQIVF="--tqivf-probes-list 10 30 99 200 400 --tqivf-reranks 100 200 2000"
TQHNSW="--tqhnsw-m 16 --tqhnsw-ef-construction 64 --tqhnsw-ef-search-list 40 100 200 --tqhnsw-reranks 100 200"
HNSW="--hnsw-m 32 --hnsw-ef-construction 128 --hnsw-ef-search 100"
WORKERS="--max-parallel-maintenance-workers 8"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Dataset keys requested on the cmdline (empty => all). Accepts several, e.g.
#   bash run_5x5_suite.sh sift1m gist1m glove200 openai1m
SELALL="$*"
run() { # run <key> ; returns 0 if this dataset should run
  [ -z "$SELALL" ] && return 0
  case " $SELALL " in *" $1 "*) return 0;; esac
  return 1
}

echo "################ 5x5 SUITE START $(ts) ################"

# ---------- agtalk (cosine, 7.74M x 768, precomputed GT) ----------
# Full search_chunks export (agtalk 7.09M + combineforum 0.64M). tqflat is an exact
# flat scan (<1 QPS at this size), so it runs at 200 queries to bound wall-clock while
# the other four run at 1000; the two CSVs are merged into 5x5_agtalk.csv.
AGT_DATA="--dataset-vectors $TQ/agtalk/agtalk_base.npy \
  --dataset-queries $TQ/agtalk/agtalk_queries.npy \
  --dataset-gt $TQ/agtalk/agtalk_gt.npy --metric cosine"
if run agtalk; then
echo "================ agtalk (cosine, 7.74M) $(ts) ================"
# (a) the four scalable methods at 1000 queries
$PY $B $AGT_DATA --methods tqivf,tqhnsw,hnsw,ivfflat --max-queries 1000 $K \
  $DB $TQIVF $TQHNSW $HNSW $WORKERS \
  --maintenance-work-mem 24GB \
  --csv $RES/5x5_agtalk_main.csv 2>&1 | tee $RES/5x5_agtalk.log
echo "agtalk main exit=${PIPESTATUS[0]} $(ts)"
# (b) tqflat (exact flat scan) at 200 queries
$PY $B $AGT_DATA --methods tqflat --max-queries 200 $K \
  $DB $TQFLAT $WORKERS \
  --maintenance-work-mem 24GB \
  --csv $RES/5x5_agtalk_tqflat.csv 2>&1 | tee -a $RES/5x5_agtalk.log
echo "agtalk tqflat exit=${PIPESTATUS[0]} $(ts)"
# (c) merge: 4-method header+body, then tqflat body (drop its header)
{ cat $RES/5x5_agtalk_main.csv; tail -n +2 $RES/5x5_agtalk_tqflat.csv; } > $RES/5x5_agtalk.csv
echo "agtalk merged -> $RES/5x5_agtalk.csv $(ts)"
fi

# ---------- SIFT1M (L2, 1M x 128, file GT) ----------
if run sift1m; then
echo "================ sift1m (L2) $(ts) ================"
$PY $B \
  --fvecs-vectors $TQ/sift/sift/sift_base.fvecs \
  --fvecs-queries $TQ/sift/sift/sift_query.fvecs \
  --ivecs-gt      $TQ/sift/sift/sift_groundtruth.ivecs \
  --metric l2 --methods $METHODS $MAXQ $K \
  $DB $TQFLAT $TQIVF $TQHNSW $HNSW $WORKERS \
  --maintenance-work-mem 16GB \
  --csv $RES/5x5_sift1m.csv 2>&1 | tee $RES/5x5_sift1m.log
echo "sift1m exit=${PIPESTATUS[0]} $(ts)"
fi

# ---------- GIST1M (L2, 1M x 960, file GT) ----------
if run gist1m; then
echo "================ gist1m (L2) $(ts) ================"
$PY $B \
  --fvecs-vectors $TQ/gist/gist/gist_base.fvecs \
  --fvecs-queries $TQ/gist/gist/gist_query.fvecs \
  --ivecs-gt      $TQ/gist/gist/gist_groundtruth.ivecs \
  --metric l2 --methods $METHODS $MAXQ $K \
  $DB $TQFLAT $TQIVF $TQHNSW $HNSW $WORKERS \
  --maintenance-work-mem 16GB \
  --csv $RES/5x5_gist1m.csv 2>&1 | tee $RES/5x5_gist1m.log
echo "gist1m exit=${PIPESTATUS[0]} $(ts)"
fi

# ---------- GloVe-200 (cosine, ~1.18M x 200, hdf5 GT) ----------
if run glove200; then
echo "================ glove200 (cosine) $(ts) ================"
$PY $B \
  --hdf5 $TQ/glove/glove-200-angular.hdf5 \
  --metric cosine --methods $METHODS $MAXQ $K \
  $DB $TQFLAT $TQIVF $TQHNSW $HNSW $WORKERS \
  --maintenance-work-mem 16GB \
  --csv $RES/5x5_glove200.csv 2>&1 | tee $RES/5x5_glove200.log
echo "glove200 exit=${PIPESTATUS[0]} $(ts)"
fi

# ---------- OpenAI-1M (cosine, 1M x 1536, precomputed GT) ----------
if run openai1m; then
echo "================ openai1m (cosine) $(ts) ================"
# GT precomputed by bench/agtalk_groundtruth.py (cosine) -> openai1m_gt.npy
$PY $B \
  --dataset-vectors $TQ/openai1m/base.npy \
  --dataset-queries $TQ/openai1m/queries.npy \
  --dataset-gt      $TQ/openai1m/openai1m_gt.npy \
  --metric cosine --methods $METHODS $MAXQ $K \
  $DB $TQFLAT $TQIVF $TQHNSW $HNSW $WORKERS \
  --maintenance-work-mem 16GB \
  --csv $RES/5x5_openai1m.csv 2>&1 | tee $RES/5x5_openai1m.log
echo "openai1m exit=${PIPESTATUS[0]} $(ts)"
fi

echo "################ 5x5 SUITE DONE $(ts) ################"
