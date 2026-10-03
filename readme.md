# Hydra-RAG: High-Precision Dual-Vector Retrieval Pipeline

A dual-vector hybrid retrieval pipeline that combines dense vector embeddings with sparse keyword search (BM25) over MS MARCO passages, deployed with a local Qdrant vector database and an interactive Streamlit benchmarking dashboard.

---

## 1. Project Overview & Architecture

Hydra-RAG addresses semantic-lexical retrieval failures in enterprise RAG systems. While standard vector search handles semantic similarity, it frequently fails on exact keywords, part numbers, or specific terminology. Conversely, naive hybrid fusion often suffers from false-positive keyword displacement, which degrades recall.

Hydra-RAG implements a **Dense-Floor Cascaded Retriever with Calibrated Lexical Fusion**:

- **Candidate Anchor (Dense Floor):** Queries the vector space using dense embeddings (`BAAI/bge-small-en-v1.5`) to preserve a **1.0000 Context Recall floor**.
- **Lexical Tie-Breaking (Sparse Leg):** Evaluates lexical term overlaps via sparse BM25 representations (`Qdrant/bm25`) within candidate horizons to boost exact keyword matches.
- **Calibrated Linear Fusion:** Employs normalized, bounded linear score fusion ($\beta = 0.035$) instead of ordinal Reciprocal Rank Fusion (RRF), preventing low-semantic distractors from displacing gold chunks while breaking semantic ties.

### Pipeline

```text
                    ┌─► Dense vector search (Qdrant, BGE-small) ─► Candidate anchor (k=5) ──┐
User query ─► Pre-retrieval                                                                 ├─► Calibrated fusion ─► Top-5 context ─► RAGAS / UI dashboard
              filtering (Qdrant)                                                            │   (beta = 0.035)
                    └─► Sparse vector search (Qdrant, BM25) ─────► Horizon (k=25) ──────────┘
```

---

## 2. Directory Structure

```text
hydra-rag/
├── app/
│   ├── streamlit_app.py        # Main entrypoint for Streamlit dashboard
│   ├── views.py                # Dashboard tab layouts and benchmark charts
│   └── ui_helpers.py           # Metric artifact loaders and data-walking utilities
├── retrieval/
│   ├── __init__.py
│   └── retriever.py            # HydraRetriever (Dense, Sparse, and Calibrated Hybrid)
├── results/
│   ├── phase1_latency.json     # Phase 1 latency metrics (p50, p95, p99, mean)
│   ├── phase2_latency.json     # Phase 2 latency metrics (p50, p95, p99, mean)
│   ├── phase1_ragas.json       # Phase 1 RAGAS baseline evaluation scores
│   └── phase2_ragas.json       # Phase 2 RAGAS hybrid evaluation scores
├── data/
│   └── corpus/                 # MS MARCO 100k+ passages subset
├── scripts/
│   ├── ingest.py               # Dataset download, embedding generation, Qdrant indexing
│   ├── benchmark_latency.py    # 100-query latency benchmarking script
│   └── evaluate_ragas.py       # RAGAS evaluation runner with Groq LLM judge
├── config.yaml                 # Qdrant client configurations and model identifiers
├── requirements.txt            # Python dependencies (free-tier only)
└── README.md                   # System documentation and execution guide
```

---

## 3. Benchmark Results & Verification

All benchmarks were evaluated on consumer hardware (AMD Ryzen 5 7530U, 12 logical cores, 16 GB RAM) on Linux against a minimum of 100,000 indexed MS MARCO passages.

### RAGAS Accuracy Evaluation (50 Queries, Groq LLM Judge)

| Metric | Acceptance Criteria | Phase 1 (Dense Baseline) | Phase 2 (Hydra Hybrid) | Status |
| --- | --- | --- | --- | --- |
| **Context Precision** | > 0.75 | **0.8598** | **0.8587** | Passed (> 0.75) |
| **Context Recall** | > 0.70 | **1.0000** | **1.0000** | Passed (1.0000 recall floor preserved) |
| **Label Recall@5** | Reference metric | **1.0000** | **1.0000** | Passed |

### Latency Benchmark (100 Consecutive Queries)

| Metric | Target Constraint | Phase 1 (Dense) | Phase 2 (Hybrid) | Delta |
| --- | --- | --- | --- | --- |
| **p50 Latency** | n/a | 29.69 ms | 37.56 ms | +7.9 ms |
| **p95 Latency** | < 300.00 ms | **37.42 ms** | **44.75 ms** | +7.3 ms |
| **p99 Latency** | n/a | 45.85 ms | 47.01 ms | +1.2 ms |
| **Mean Latency** | n/a | 30.59 ms | 38.09 ms | +7.5 ms |

---

## 4. Setup & Installation

### Prerequisites

- Python 3.10+
- Docker (for the local Qdrant instance) or a native Qdrant binary
- Consumer CPU/system with ≥ 8 GB RAM
- Free-tier Groq API key (for RAGAS evaluation only)

### 1. Clone Repository & Set Up Virtual Environment

```bash
git clone https://github.com/your-org/hydra-rag.git
cd hydra-rag

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch Vector Database (Qdrant)

Run Qdrant via Docker locally on port `6333`:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    --name qdrant_hydra qdrant/qdrant:latest
```

### 3. Ingest MS MARCO Dataset (≥ 100k Passages)

Downloads the MS MARCO passage dataset from Hugging Face, generates dense embeddings (`BAAI/bge-small-en-v1.5`) and sparse BM25 representations (`Qdrant/bm25`), and upserts vectors and metadata into Qdrant:

```bash
python scripts/ingest.py --limit 100000
```

---

## 5. Running Evaluations & Benchmarks

### Latency Benchmark (100 Queries)

Runs 100 sequential queries across dense and hybrid configurations, capturing p50, p95, p99, and stage breakdowns:

```bash
python scripts/benchmark_latency.py --queries 100
```

Outputs are stored in `results/phase1_latency.json` and `results/phase2_latency.json`.

### RAGAS Accuracy Evaluation

Evaluates Context Precision and Context Recall across sample queries using the Groq free-tier LLM judge:

```bash
export GROQ_API_KEY="your_groq_key_here"
python scripts/evaluate_ragas.py --phase 1
python scripts/evaluate_ragas.py --phase 2
```

Outputs are written to `results/phase1_ragas.json` and `results/phase2_ragas.json`.

---

## 6. Interactive Demo & Streamlit Dashboard

Launch the benchmarking dashboard and query testing interface:

```bash
streamlit run app/streamlit_app.py
```

### Feature Checklist in Dashboard

- **Live Query Interface:** Toggle between Phase 1 (Dense) and Phase 2 (Hybrid) retrieval modes to inspect returned passages, scores, and metadata.
- **Metadata Pre-Filtering:** Test structured pre-filtering on indexed attributes directly at the Qdrant storage level.
- **Latency Visualizations:** Inspect p50, p95, p99 metrics and stage breakdowns (`embed_ms`, `search_ms`, `fuse_ms`) across 100-query distributions.
- **Live Index Modifications:** Verify runtime document upsertion and deletion by ID without re-indexing the corpus.

---

## 7. Compliance Matrix

| Requirement ID | Specification | Implementation Verification | Status |
| --- | --- | --- | --- |
| **FR-1** | Ingest ≥ 100,000 MS MARCO passages | Indexed into Qdrant with dense vectors, BM25, and payloads | Compliant |
| **FR-2** | Baseline Dense Retrieval & RAGAS | BGE-small cosine search with logged baseline precision and recall | Compliant |
| **FR-3** | Hybrid Search with Configurable Fusion | Dense-anchored BM25 score fusion with configurable β weighting | Compliant |
| **FR-4** | Metadata Filtering | Pre-retrieval filtering executed at Qdrant payload index level | Compliant |
| **FR-5** | Live Index Updates | Dedicated point upsert and point delete methods via Qdrant Client | Compliant |
| **FR-6** | Query Interface | Interactive Streamlit application supporting mode toggling and latency views | Compliant |
| **NFR-1 / 2** | Context Precision > 0.75, Recall > 0.70 | Precision: **0.8587**, Recall: **1.0000** | Compliant |
| **NFR-3** | Query Latency p95 < 300 ms | Measured hybrid p95: **44.75 ms** | Compliant |
| **C-01** | Free-Tier Tools Only | FastEmbed, local Qdrant, Groq free-tier LLM judge | Compliant |
