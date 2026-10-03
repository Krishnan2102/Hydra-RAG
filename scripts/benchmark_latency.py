import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import yaml

from retrieval.retriever import HydraRetriever


def get_hardware_info():
    info = {
        "os": platform.system(),
        "platform_release": platform.release(),
        "cpu": platform.processor() or "Unknown",
        "cores_logical": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    info["ram_gb"] = round(int(line.split(":")[1].split()[0]) / (1024**2), 2)
                    break
    except Exception:
        pass
    return info


def calc_percentiles(arr):
    return {
        "mean": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "hybrid"
    if mode not in ["dense", "hybrid"]:
        print(f"Unknown mode '{mode}'. Defaulting to 'hybrid'.")
        mode = "hybrid"

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_path = Path(cfg["paths"]["eval_set"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    n_warmup = cfg.get("benchmark", {}).get("warmup_queries", 10)
    n_queries = cfg.get("benchmark", {}).get("latency_queries", 100)

    print(f"Loading queries from {eval_path}...")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if isinstance(eval_data, list):
        queries = [q["query"] if isinstance(q, dict) else q for q in eval_data]
    elif isinstance(eval_data, dict) and "queries" in eval_data:
        queries = [q["query"] if isinstance(q, dict) else q for q in eval_data["queries"]]
    else:
        raise ValueError("Could not parse queries from eval_set.json")

    if len(queries) < (n_warmup + n_queries):
        multiplier = ((n_warmup + n_queries) // len(queries)) + 1
        queries = (queries * multiplier)[: (n_warmup + n_queries)]

    warmup_set = queries[:n_warmup]
    bench_set = queries[n_warmup : n_warmup + n_queries]

    retriever = HydraRetriever()

    print(f"\nWarming up engine ({n_warmup} queries in {mode.upper()} mode)...")
    for q in warmup_set:
        if mode == "dense":
            retriever.search_dense(q, limit=5)
        else:
            retriever.search_hybrid(q, limit=5)

    print(f"\nExecuting official {mode.upper()} benchmark ({n_queries} consecutive queries)...")
    total_times = []
    dense_leg_times = []
    sparse_leg_times = []
    fusion_times = []

    for i, q in enumerate(bench_set, 1):
        if mode == "dense":
            res = retriever.search_dense(q, limit=5)
            total_times.append(res["timings_ms"]["total_ms"])
        else:
            res = retriever.search_hybrid(q, limit=5)
            total_times.append(res["timings_ms"]["total_ms"])
            dense_leg_times.append(res["timings_ms"]["dense_leg_ms"])
            sparse_leg_times.append(res["timings_ms"]["sparse_leg_ms"])
            fusion_times.append(res["timings_ms"]["fusion_ms"])

        if i % 25 == 0:
            print(f"  Processed {i}/{n_queries} queries | Current p95: {np.percentile(total_times, 95):.2f} ms")

    bench_results = {
        "phase": f"Phase {'1: Dense Baseline' if mode == 'dense' else '2: Hybrid Search'}",
        "mode": mode,
        "query_count": n_queries,
        "hardware": get_hardware_info(),
        "metrics_ms": {
            "total_latency": calc_percentiles(total_times),
        },
    }

    if mode == "hybrid":
        bench_results["metrics_ms"]["dense_leg"] = calc_percentiles(dense_leg_times)
        bench_results["metrics_ms"]["sparse_leg"] = calc_percentiles(sparse_leg_times)
        bench_results["metrics_ms"]["fusion"] = calc_percentiles(fusion_times)

    out_file = results_dir / f"phase2_latency_benchmark.json" if mode == "hybrid" else results_dir / "phase1_latency_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  {bench_results['phase'].upper()} LATENCY BENCHMARK (N = {n_queries})")
    print("=" * 60)
    tot = bench_results["metrics_ms"]["total_latency"]
    print(f"Total Latency:  p50={tot['p50']}ms | p90={tot['p90']}ms | p95={tot['p95']}ms | p99={tot['p99']}ms")
    if mode == "hybrid":
        d_leg = bench_results["metrics_ms"]["dense_leg"]
        s_leg = bench_results["metrics_ms"]["sparse_leg"]
        fus = bench_results["metrics_ms"]["fusion"]
        print(f"Dense Leg:      p50={d_leg['p50']}ms | p95={d_leg['p95']}ms")
        print(f"Sparse Leg:     p50={s_leg['p50']}ms | p95={s_leg['p95']}ms")
        print(f"Fusion Time:    p50={fus['p50']}ms | p95={fus['p95']}ms")
    print("=" * 60)
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()