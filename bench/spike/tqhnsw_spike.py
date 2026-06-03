"""TQ+HNSW viability spike: scoring + traversal (A/B/C) + verdict.  Self-contained;
runs on the-fire.  Outputs a CSV and bench/spike/RESULTS.md.  The GREEN/YELLOW/RED
bands are a framing lens — the raw numbers are always written for maintainer review."""
import argparse
import csv
import os
import numpy as np

import datasets as D
import tq_numpy as TQ
import scoring as S
import traversal as T

DATASETS = ["sift1m", "glove200", "openai1m"]


def min_ef_to_threshold(curve, baseline, frac=0.98):
    """Smallest efSearch whose recall >= frac*baseline (or None)."""
    for ef in sorted(curve):
        if curve[ef] >= frac * baseline:
            return ef
    return None


def verdict(full, best_variant_curve, efs):
    """GREEN/YELLOW/RED per spec section 7 - advisory only."""
    base_ef = 100
    baseline = full[base_ef]
    ef = min_ef_to_threshold(best_variant_curve, baseline)
    if ef is not None and ef <= 2 * base_ef:
        return "GREEN", baseline, ef
    if ef is not None:
        return "YELLOW", baseline, ef
    return "RED", baseline, None


def run_dataset(key, n_queries):
    d = D.load(key, n_queries=n_queries)
    dim = d["base"].shape[1]
    m = TQ.TQModel(dim)
    rec = S.quantized_search_vectors(m.reconstruct(d["base"]), d["metric"])
    base, q, gt, metric = d["base"], d["queries"], d["gt"], d["metric"]

    # scoring (graph-free)
    gate = S.bruteforce_rerank_recall(base, rec, q, gt, metric, rerank=200)
    curve = S.overfetch_curve(base, rec, q, gt, metric)
    fid = S.near_neighbor_rank_fidelity(base, rec, q, metric)

    # traversal
    full = T.variant_full(base, q, gt, metric)
    A = T.variant_A_all_quantized(base, rec, q, gt, metric)
    B = T.variant_B_build_full_traverse_quant(base, rec, q, gt, metric)
    C = T.variant_C_faiss_sq(base, q, gt, metric)

    # verdict uses the better of A and B at each ef
    best = {ef: max(A[ef], B[ef]) for ef in A}
    band, baseline, ef = verdict(full, best, T.EFS)

    return dict(key=key, dim=dim, metric=metric, gate=gate, curve=curve, fid=fid,
                full=full, A=A, B=B, C=C, band=band, baseline=baseline, ef_to_98=ef)


def write_outputs(rows, csv_path, md_path):
    # ---- CSV: one row per (dataset, variant, ef) ----
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "dim", "metric", "variant", "efSearch", "recall@10"])
        for r in rows:
            for ef in T.EFS:
                w.writerow([r["key"], r["dim"], r["metric"], "full", ef, f"{r['full'][ef]:.4f}"])
                w.writerow([r["key"], r["dim"], r["metric"], "A_allquant", ef, f"{r['A'][ef]:.4f}"])
                w.writerow([r["key"], r["dim"], r["metric"], "B_buildfull", ef, f"{r['B'][ef]:.4f}"])
                w.writerow([r["key"], r["dim"], r["metric"], "C_faisssq", ef, f"{r['C'][ef]:.4f}"])
    # ---- Markdown ----
    L = ["# TQ + HNSW viability spike - results", "",
         "200k subset per dataset; HNSW M=32 efC=128. Verdict bands are advisory "
         "(spec section 7); raw numbers below are the deliverable. Quantized distance uses "
         "distance-to-reconstruction, a conservative proxy for the asymmetric ADC.", ""]
    for r in rows:
        L += [f"## {r['key']} ({r['dim']}d, {r['metric']}) - **{r['band']}**", "",
              f"- numpy-TQ gate (rerank=200 brute force): **{r['gate']:.4f}**",
              f"- near-neighbor rank fidelity: Kendall-tau={r['fid']['kendall_tau']:.3f}, "
              f"Spearman={r['fid']['spearman']:.3f}",
              f"- over-fetch curve recall@10: " +
              ", ".join(f"C={c}:{v:.3f}" for c, v in r['curve'].items()),
              f"- full-precision baseline recall@10 (ef=100): {r['baseline']:.4f}",
              f"- min efSearch reaching 0.98xbaseline (best of A/B): {r['ef_to_98']}", "",
              "| efSearch | full | A all-quant | B build-full | C faiss-SQ |",
              "|---|---|---|---|---|"]
        for ef in T.EFS:
            L.append(f"| {ef} | {r['full'][ef]:.4f} | {r['A'][ef]:.4f} | "
                     f"{r['B'][ef]:.4f} | {r['C'][ef]:.4f} |")
        L.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "tqhnsw_spike.csv"))
    ap.add_argument("--md", default=os.path.join(os.path.dirname(__file__), "RESULTS.md"))
    args = ap.parse_args()

    rows = []
    for key in args.datasets.split(","):
        print(f"=== {key} ===", flush=True)
        r = run_dataset(key, args.n_queries)
        print(f"  {key}: band={r['band']} gate={r['gate']:.4f} "
              f"baseline={r['baseline']:.4f} ef98={r['ef_to_98']}", flush=True)
        rows.append(r)
    write_outputs(rows, args.csv, args.md)
    print(f"\nWrote {args.csv} and {args.md}")


if __name__ == "__main__":
    main()
