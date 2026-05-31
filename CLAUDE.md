# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

pgvector is a PostgreSQL extension (C + SQL) that adds vector data types and approximate-nearest-neighbor index access methods. It is built with PGXS, the standard Postgres extension toolchain — there is no separate build system.

## Build & install

```sh
make              # compiles src/*.c into vector.so, builds sql/vector--0.8.2.sql
make install      # copies the .so, control file, and SQL into the Postgres install tree
```

`make` shells out to `pg_config` to locate the server headers and PGXS. Build against a specific Postgres install by overriding it: `make PG_CONFIG=/path/to/pg_config`. For a portable (non-`-march=native`) build use `make OPTFLAGS=""` — the default `-march=native` plus auto-vectorization flags (`-ftree-vectorize -fassociative-math ...`) are what let the distance kernels vectorize, so build flags materially affect performance and numeric results.

`EXTVERSION` in the `Makefile` is the source of truth for the current version. The built `sql/vector--$(EXTVERSION).sql` is just a copy of `sql/vector.sql` — keep those two in sync.

## Tests

Two independent suites:

```sh
make installcheck        # pg_regress: SQL-level regression tests (test/sql/*.sql vs test/expected/*.out)
make prove_installcheck  # TAP tests: recall/WAL/vacuum/crash behavior (test/t/*.pl, Perl)
```

Run a single test from each:

```sh
make installcheck REGRESS=hnsw_vector                            # one regression test (no .sql suffix)
make prove_installcheck PROVE_TESTS=test/t/001_ivfflat_wal.pl    # one TAP test
```

Both targets run against a **running Postgres instance** with the extension already `make install`ed — `installcheck` (not `check`) means "test the installed extension," so rebuild and reinstall before testing C changes. A regression test is a pair: `test/sql/foo.sql` (input) and `test/expected/foo.out` (expected output); a mismatch fails the test, and intentional output changes require updating the `.out` file.

The TAP tests (`test/t/`) are where correctness of the *index algorithms* lives — they build indexes over random data and assert recall thresholds, exercise WAL replay, vacuum, and parallel build. Pure SQL/type changes are covered by the regression suite; index-behavior changes need the TAP suite.

### Debug build flags

Rebuild with `make clean && PG_CFLAGS="..." make && make install`:

- `-DUSE_ASSERT_CHECKING` — enable assertions (use this when touching index internals)
- `-DHNSW_MEMORY` / `-DIVFFLAT_MEMORY` — report index build memory usage
- `-DIVFFLAT_BENCH` — IVFFlat build timing
- `-DIVFFLAT_KMEANS_DEBUG` — k-means convergence metrics

## Architecture

The extension has three layers: **vector data types**, **two index access methods**, and a **SIMD distance-dispatch** mechanism that connects them. The SQL surface (types, operators, operator classes, AM handlers) is defined entirely in `sql/`; the C in `src/` implements it.

### Data types (`src/`)

Four vector representations, each a self-contained type with in/out/recv/send, casts, comparison, and distance functions:

- `vector.c` — dense `float4` (the primary type)
- `halfvec.c` — half-precision `float2`/fp16; relies on `halfutils.c` for fp16↔fp32 conversion (`HalfToFloat4` is the single most-connected function in the codebase — nearly everything fp16 routes through it)
- `sparsevec.c` — sparse `{indices, values}` representation
- `bitvec.c` — binary vectors; `bitutils.c` implements Hamming/Jaccard

Casts form a mesh between vector/halfvec/sparsevec, so a change to one type's cast or distance contract usually has counterparts in the others.

### Distance dispatch (`halfutils.c`, `bitutils.c`)

Distance kernels are selected at extension load, not compile time. `_PG_init` → `HalfvecInit`/`BitvecInit` probe CPU features (`SupportsAvx512Popcount`, `SupportsCpuFeature` for F16C) and set **function pointers** (`HalfvecL2SquaredDistance`, `BitHammingDistance`, etc.) to either a `*Default` scalar variant or an `*Avx512Popcount`/`*F16c` SIMD variant. The SIMD variants tail-call the Default variant for the remainder elements. When adding a distance function, you add the dispatch pointer + Default variant; SIMD variants are optional and must stay numerically consistent with Default.

### Index access methods

Both implement the standard Postgres index AM interface (`ambuild`, `aminsert`, `ambeginscan`/`amgettuple`/`amrescan`/`amendscan`, `ambulkdelete`/`amvacuumcleanup`, `amcostestimate`, `amoptions`). The AM handler functions (`hnswhandler`, `ivfflathandler`) are the entry points wired up in SQL; everything else hangs off them.

**HNSW** (`hnsw*.c`) — Hierarchical Navigable Small World graph. The defining design choice is a **two-phase build**: `hnswbuild.c` builds the whole graph in memory first (relative pointers so it can live in shared memory for parallel build), then flushes to disk pages. Because of this, nearly every graph-mutation operation exists in **two mirrored variants** — in-memory (build) and on-disk (insert/vacuum-repair) — unified through a `base` pointer abstraction so the same core (`HnswSearchLayer` = paper Algorithm 2, `SelectNeighbors` = Algorithm 4 heuristic pruning, `HnswUpdateConnection`) is reused across all three paths:
  - `hnswbuild.c` — in-memory graph construction + parallel build coordination
  - `hnswinsert.c` — on-disk single-tuple insert
  - `hnswscan.c` — query: greedy layer search + iterative-scan resume
  - `hnswvacuum.c` — three-pass vacuum (remove heap TIDs → repair graph → mark deleted), order matters for concurrent correctness
  - `hnswutils.c` — the shared graph algorithms and page-layout helpers

**IVFFlat** (`ivf*.c`) — inverted file with k-means clustering. Build pipeline is sample → cluster → assign: `ivfbuild.c` reservoir-samples rows (`SampleRows`, bounded sample not the full table), runs `ivfkmeans.c` (Elkan's algorithm using the triangle inequality to prune distance calcs, k-means++ seeding), assigns each tuple to its nearest centroid/list, and writes list pages. `ivfscan.c` picks the nearest `probes` lists and scans only those tuples — `probes` vs `lists` is the recall/speed knob. `ivfinsert.c`, `ivfvacuum.c` handle incremental insert and vacuum.

Both AMs share a type-dispatch vtable pattern (`HnswGetTypeInfo` / `IvfflatGetTypeInfo`) so one index implementation works across all four data types.

### SQL & versioning (`sql/`)

`sql/vector.sql` is the full current definition (types, the six distance operators `<->` L2 / `<#>` inner product / `<=>` cosine / `<+>` L1 / `<~>` Hamming / `<%>` Jaccard, operator classes per (type × AM), aggregates, AM handlers). The `sql/vector--X.Y.Z--A.B.C.sql` files are an **append-only chain of upgrade migrations** — `ALTER EXTENSION vector UPDATE` walks them in sequence. Adding a SQL object means: add it to `vector.sql` **and** add a new `--old--new.sql` migration; never edit a released migration file. The bulk of the modern surface (halfvec, sparsevec, bit operators) landed in `vector--0.6.2--0.7.0.sql`.

## Conventions

- C is formatted with `pgindent` (Postgres's formatter) — match the surrounding tab-indented, brace-on-own-line style; commits like "Ran latest pgindent" reformat en masse.
- `[skip ci]` is used on formatting/comment-only commits.
