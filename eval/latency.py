import sys
import json
import numpy as np
from pathlib import Path

# Append project root to system path for module resolution
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from retrieval.dense import search

def main():
    eval_file = Path("data/eval_set.json")
    out_file = Path("results/phase1_latency.json")
    
    with open(eval_file, "r", encoding="utf-8") as f:
        queries = [q["query"] for q in json.load(f)]
        
    # Enforce exactly 100 iterations per the benchmark requirement
    if len(queries) < 100:
        queries = (queries * (100 // len(queries) + 1))[:100]
    else:
        queries = queries[:100]

    print("Executing hardware warm-up sequence (5 iterations)...")
    for q in queries[:5]:
        _ = search(query=q, mode="dense", top_k=5)
        
    print("Initiating 100-query latency benchmark...")
    embed_times = []
    search_times = []
    total_times = []
    
    for q in queries:
        results = search(query=q, mode="dense", top_k=5)
        if results:
            t = results[0]["timings"]
            embed_times.append(t["embed_ms"])
            search_times.append(t["search_ms"])
            total_times.append(t["total_ms"])

    # Compute statistical percentiles
    metrics = {
        "p50_total_ms": round(np.percentile(total_times, 50), 2),
        "p95_total_ms": round(np.percentile(total_times, 95), 2),
        "p99_total_ms": round(np.percentile(total_times, 99), 2),
        "avg_embed_ms": round(float(np.mean(embed_times)), 2),
        "avg_search_ms": round(float(np.mean(search_times)), 2),
        "avg_total_ms": round(float(np.mean(total_times)), 2)
    }
    
    print("\n--- Latency Telemetry ---")
    print(json.dumps(metrics, indent=2))
    
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nLatency telemetry persisted to {out_file}")

if __name__ == "__main__":
    main()