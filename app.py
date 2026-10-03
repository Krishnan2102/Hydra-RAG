import sys
import json
from pathlib import Path
import numpy as np
import streamlit as st

# Append project root to system path for module resolution
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from retrieval.dense import search

st.set_page_config(page_title="HYDRA-RAG", layout="wide")

st.title("HYDRA-RAG: High-Precision Retrieval")
st.markdown("Phase 1: Baseline Dense Retrieval Telemetry")

# Initialize session state telemetry buffer
if "latency_history" not in st.session_state:
    st.session_state.latency_history = {
        "embed_ms": [],
        "search_ms": [],
        "total_ms": []
    }

# UI Input Controls
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter natural language query:")
with col2:
    mode = st.selectbox(
        "Retrieval Mode",
        options=["Dense", "Hybrid (BM25 + Dense - Phase 2)"],
        index=0
    )

# Sidebar: Distribution Telemetry & Benchmark Inspection
with st.sidebar:
    st.header("Latency Profiling")
    
    # Session-level distribution metrics
    total_runs = len(st.session_state.latency_history["total_ms"])
    st.caption(f"Active Session Query Count: {total_runs}")
    
    if total_runs > 0:
        totals = np.array(st.session_state.latency_history["total_ms"])
        p50 = np.percentile(totals, 50)
        p95 = np.percentile(totals, 95)
        p99 = np.percentile(totals, 99)
        
        st.subheader("Session Quantiles")
        st.metric(label="p50 (Median)", value=f"{p50:.2f} ms")
        st.metric(label="p95 Latency", value=f"{p95:.2f} ms")
        st.metric(label="p99 Latency", value=f"{p99:.2f} ms")
        
        if st.button("Flush Telemetry Buffer"):
            st.session_state.latency_history = {"embed_ms": [], "search_ms": [], "total_ms": []}
            st.rerun()
    else:
        st.info("Execute queries to populate session quantile distribution.")

    # Static Phase 1 Benchmark Metrics from disk (eval/latency.py output)
    benchmark_file = Path("results/phase1_latency.json")
    if benchmark_file.exists():
        st.divider()
        st.subheader("Logged Phase 1 Baseline")
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)
        st.json(bench_data)

if query:
    st.divider()
    
    with st.spinner("Executing dense HNSW traversal..."):
        results = search(query=query, mode="dense", top_k=5)
    
    if results:
        latency = results[0]["timings"]
        
        # Append telemetry to session buffer
        st.session_state.latency_history["embed_ms"].append(latency["embed_ms"])
        st.session_state.latency_history["search_ms"].append(latency["search_ms"])
        st.session_state.latency_history["total_ms"].append(latency["total_ms"])
        
        # Render instantaneous single-query latency breakdown
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Query Vectorization", f"{latency['embed_ms']} ms")
        col_m2.metric("Qdrant HNSW Search", f"{latency['search_ms']} ms")
        col_m3.metric("Total Latency", f"{latency['total_ms']} ms")
        
        # Display Retrieved Candidates
        st.subheader("Retrieved Candidate Passages")
        for i, res in enumerate(results):
            with st.expander(f"Rank {i+1} | Score: {res['score']:.4f} | Source: {res['source']}", expanded=(i == 0)):
                st.write(res["text"])
                st.caption(f"Passage ID: {res['id']} | Category Tag: {res['category']}")
    else:
        st.warning("Empty result set returned.")