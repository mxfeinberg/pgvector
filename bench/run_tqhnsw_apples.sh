#!/usr/bin/env bash
# Apples-to-apples tqhnsw tuning on agtalk (7.74M x 768, cosine).
#
# The 5x5 suite ran tqhnsw at its DEFAULTS (m=16, ef_construction=64) against hnsw at
# m=32/ef_construction=128 — double the graph connectivity and build beam. This rebuilds
# tqhnsw at the SAME m=32/ef_construction=128 as hnsw, and sweeps ef_search wider (the 5x5
# stopped at 200 while recall was still climbing) plus a rerank sweep down to 30 (recall was
# flat 100->200, so rerank looked over-provisioned — lower rerank = fewer heap fetches = QPS).
#
# Build-once, query-many: one m=32 index, then ef_search {100,200,400,800} x rerank {30,50,100}.
#   bash bench/run_tqhnsw_apples.sh
set -u
cd ~/pgvector || exit 1

PY=python3
B=bench/tqflat_bench.py
RES=bench/results
TQ=~/tqdata
mkdir -p "$RES"

DB="--user max --host /var/run/postgresql --dbname tqtest"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "################ tqhnsw apples-to-apples START $(ts) ################"

$PY $B \
  --dataset-vectors $TQ/agtalk/agtalk_base.npy \
  --dataset-queries $TQ/agtalk/agtalk_queries.npy \
  --dataset-gt      $TQ/agtalk/agtalk_gt.npy \
  --metric cosine --methods tqhnsw --max-queries 1000 --k 10 \
  $DB \
  --tqhnsw-m 32 --tqhnsw-ef-construction 128 \
  --tqhnsw-ef-search-list 100 200 400 800 \
  --tqhnsw-reranks 30 50 100 \
  --max-parallel-maintenance-workers 8 --maintenance-work-mem 24GB \
  --csv $RES/agtalk_tqhnsw_m32.csv 2>&1 | tee $RES/agtalk_tqhnsw_m32.log
echo "tqhnsw m32 exit=${PIPESTATUS[0]} $(ts)"
echo "################ tqhnsw apples-to-apples DONE $(ts) ################"
