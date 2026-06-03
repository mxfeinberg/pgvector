#!/usr/bin/env bash
# TQ-HNSW vs HNSW benchmark suite (the full sweep for sub-project #5's payoff).
# Runs SIFT1M (L2), GloVe-200 (cosine), OpenAI-1M (cosine) SEQUENTIALLY so that
# concurrent index builds / loads don't contaminate the latency measurements.
# Each dataset writes its own CSV + log so partial results survive a failure.
#
# Mirrors bench/run_tqivf_suite.sh. tqhnsw builds ONCE per dataset and is queried
# across the ef_search × rerank grid (those are query-time GUCs); hnsw is the
# matched-m/ef_construction reference baseline.
#
# PREREQS (see the handoff 2026-06-03-tqhnsw-blockscan-done-next-benchmark-handoff
# .md and the bench-build handoff):
#   1. the-fire `vector.so` SHOULD be built WITHOUT -DUSE_ASSERT_CHECKING for
#      representative build/latency numbers (index size + recall are unaffected by
#      asserts). Rebuild + `sudo make install` first if the current .so is the
#      assert build.
#   2. The `tqtest` DB's `vector` extension MUST include the tqhnsw AM. If a fresh
#      DB or a stale one: `psql -d tqtest -c "DROP EXTENSION vector CASCADE; CREATE
#      EXTENSION vector;"` (verify with: SELECT amname FROM pg_am WHERE amname LIKE
#      'tq%'; -> must list tqhnsw). The persistent tqtest predated tqhnsw.
set -u
cd ~/pgvector || exit 1

PY=python3
B=bench/tqflat_bench.py
RES=bench/results
mkdir -p "$RES"

DB="--user max --host /var/run/postgresql --dbname tqtest"
# tqhnsw built once per dataset (m/ef_construction), queried across this grid:
TQHNSW="--tqhnsw-m 16 --tqhnsw-ef-construction 64 \
        --tqhnsw-ef-search-list 40 100 200 --tqhnsw-reranks 100 200 \
        --tqhnsw-force-scalar off"
# hnsw matched baseline (same m / ef_construction; single ef_search reference).
HNSW="--hnsw-m 16 --hnsw-ef-construction 64 --hnsw-ef-search 100"
MEM="--maintenance-work-mem 16GB"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "================ SIFT1M (L2) $(ts) ================"
$PY $B \
  --fvecs-vectors ~/tqdata/sift/sift/sift_base.fvecs \
  --fvecs-queries ~/tqdata/sift/sift/sift_query.fvecs \
  --ivecs-gt      ~/tqdata/sift/sift/sift_groundtruth.ivecs \
  --metric l2 --no-exact --max-queries 500 \
  --methods tqhnsw,hnsw \
  $DB $TQHNSW $HNSW $MEM \
  --csv $RES/sift1m_tqhnsw_x86.csv 2>&1 | tee $RES/sift1m_tqhnsw_x86.log
echo "SIFT1M exit=${PIPESTATUS[0]} $(ts)"

echo "================ GloVe-200 (cosine) $(ts) ================"
$PY $B \
  --hdf5 ~/tqdata/glove/glove-200-angular.hdf5 \
  --metric cosine --no-exact --max-queries 500 \
  --methods tqhnsw,hnsw \
  $DB $TQHNSW $HNSW $MEM \
  --csv $RES/glove200_tqhnsw_x86.csv 2>&1 | tee $RES/glove200_tqhnsw_x86.log
echo "GloVe exit=${PIPESTATUS[0]} $(ts)"

echo "================ OpenAI-1M (cosine) $(ts) ================"
# The headline: 1536-d is where the ~8x memory win + the block-kernel latency win
# should be largest. tqhnsw build here is the slow part (single-threaded; parallel
# build is sub-project #3, deferred) -- expect tens of minutes.
$PY $B \
  --dataset-vectors ~/tqdata/openai1m/base.npy \
  --dataset-queries ~/tqdata/openai1m/queries.npy \
  --metric cosine \
  --methods tqhnsw,hnsw \
  $DB $TQHNSW $HNSW $MEM \
  --csv $RES/openai1m_tqhnsw_x86.csv 2>&1 | tee $RES/openai1m_tqhnsw_x86.log
echo "OpenAI1M exit=${PIPESTATUS[0]} $(ts)"

echo "================ SUITE DONE $(ts) ================"
