# HYDRA-RAG: Two-Stage Hybrid Search & Evaluation Platform

HYDRA-RAG is a high-performance retrieval and evaluation platform built on **Qdrant**, **FastEmbed**, and **RAGAS**. The system implements a two-stage hybrid retrieval architecture:

* **Phase 1 (Dense Baseline):** Dense semantic search using deep representation models (`BAAI/bge-small-en-v1.5`).
* **Phase 2 (Hybrid + Re-ranking):** Multi-channel candidate retrieval combining dense semantic embeddings with sparse BM25 token vectors, followed by neural re-ranking using a cross-encoder (`BAAI/bge-reranker-base`).

The platform includes end-to-end data ingestion pipelines, latency benchmarking, automated RAGAS quality evaluation, and an interactive **Streamlit** dashboard for side-by-side comparison and telemetry inspection.

---

## Architecture Overview

```text
                         +-----------------------+
                         |      User Query       |
                         +-----------+-----------+
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
       +--------------------+                +--------------------+
       |   Dense Embedder   |                |   Sparse Embedder  |
       | (BGE Small / ONNX) |                |   (BM25 Tokenizer) |
       +----------+---------+                +----------+---------+
                  |                                     |
                  v                                     v
       +--------------------+                +--------------------+
       | Qdrant Dense Index |                | Qdrant Sparse Leg  |
       +----------+---------+                +----------+---------+
                  |                                     |
                  +------------------+------------------+
                                     |
                                     v
                         +-----------------------+
                         |  Candidate Pool Union |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Neural Cross-Encoder  |
                         |   (bge-reranker-base) |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         |  Top-K Ranked Passages|
                         +-----------------------+
```

---

## Repository Structure

```text
├── app/                      # Streamlit dashboard and UI logic
│   ├── config.py             # UI configuration and state management
│   ├── streamlit_app.py      # Dashboard entry point
│   ├── ui_helpers.py         # Visual components, charts, and metrics loaders
│   └── views.py              # Search page and Phase comparison views
├── data/                     # Corpus files, ground truth, and precomputed embeddings
│   ├── categories.json       # Corpus topic taxonomy
│   ├── embeddings.npy        # Serialized dense vector arrays
│   ├── eval_set.json         # Labelled questions and ground-truth contexts
│   └── passages.jsonl        # Raw passage documents (MS MARCO slice)
├── eval/                     # Evaluation harnesses
│   ├── __init__.py
│   └── eval_ragas.py         # RAGAS evaluation runner (Precision, Recall, etc.)
├── fusion/                   # Fusion algorithms
│   └── rank_fusion.py        # Reciprocal Rank Fusion (RRF) and merge utilities
├── ingest/                   # Document preprocessing and indexing pipelines
│   ├── categories.py         # Category classification and extraction logic
│   ├── embed.py              # FastEmbed batch vector computation
│   ├── explore_msmarco.py    # Dataset exploration and schema checking
│   ├── index.py              # High-level index orchestration
│   ├── index_service.py      # Background ingestion worker service
│   ├── prepare_data.py       # Corpus cleaning and formatting
│   └── qdrant_ingest.py      # Batch payload writer for Qdrant collection
├── qdrant_storage/           # Local Qdrant persistent storage volume
├── results/                  # Serialized evaluation & latency telemetry JSONs
│   ├── phase1_latency.json
│   ├── phase1_ragas.json
│   ├── phase2_latency.json
│   ├── phase2_ragas.json
│   └── ...
├── retrieval/                # Core retrieval engine
│   ├── __init__.py
│   └── retriever.py          # HydraRetriever: multi-stage retrieval & re-ranking
├── scripts/                  # Diagnostics and benchmark CLI tools
│   ├── benchmark_latency.py  # Automated p50, p95, p99 latency test suite
│   ├── peek_data.py          # Inspect records from passages.jsonl
│   ├── smoke_test.py         # End-to-end operational pipeline healthcheck
│   ├── test_live_updates.py  # Test dynamic ingestion into Qdrant
│   └── test_retriever.py     # Unit test runner for retriever components
├── config.yaml               # System parameters (models, vectors, Qdrant ports)
├── docker-compose.yml        # Qdrant engine container specification
├── requirements.txt          # Python dependencies
└── .env                      # API keys and local environment variables
```
## Benchmark & Evaluation Results

The system was benchmarked across **100,000+ indexed passages** (`data/passages.jsonl`) using the golden evaluation set (`data/eval_set.json`). 

Performance was tracked across two tracks:
1. **Phase 1 Baseline (Dense):** Pure vector retrieval using `BAAI/bge-small-en-v1.5`.
2. **Phase 2 Hybrid + Rerank:** Dual-leg retrieval (Dense + BM25 Sparse) with `BAAI/bge-reranker-base` cross-encoder reranking over candidate pools.

---

### 1. RAGAS Quality Metrics

Quality was evaluated across labelled queries with the target of beating the hackathon accuracy threshold without LLM hallucination:

| Metric | Phase 1 (Dense) | Phase 2 (Hybrid + Rerank) | Target Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision** | ~0.742 | **0.884** | `> 0.800` | **Met** |
| **Context Recall** | ~0.691 | **0.865** | `> 0.750` | **Met** |
| **MRR @ 5** | 0.718 | **0.872** | — | Improved (+21.4%) |
| **Hit Rate @ 5** | 0.812 | **0.946** | — | Improved (+16.5%) |

> **Key Finding:** Adding lexical BM25 token vectors eliminated misses on named entities, specific numeric codes, and verbatim phrasing, while the cross-encoder lifted precision by pruning false-positive semantic matches.

---

### 2. Latency Benchmarks (CPU Inference)

Evaluated under multi-threaded CPU execution (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`) to simulate constrained production hosting without dedicated GPU instances:

| Metric | Phase 1 (Dense) | Phase 2 (Hybrid + Rerank) | Hackathon Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **p50 Latency** | 22.4 ms | 68.5 ms | — | — |
| **p95 Latency** | 41.8 ms | **128.2 ms** | `< 300.0 ms` | **Met** |
| **p99 Latency** | 56.1 ms | 148.7 ms | — | — |
| **Mean Latency** | 24.8 ms | 74.1 ms | — | — |

---

### 3. Stage-by-Stage Latency Breakdown (Phase 2 Hybrid)

To keep overall p95 latency under the strict **300 ms** target budget, candidate pools and text truncation were tuned per stage:

| Stage | Component / Engine | Average Latency | Description |
| :--- | :--- | :---: | :--- |
| **Dense Embedding** | FastEmbed (ONNX) | ~14.2 ms | Generates 384-d dense vector |
| **Sparse Tokenization**| FastEmbed BM25 | ~3.8 ms | Computes token indices and weights |
| **Qdrant Search** | Vector DB Engine | ~18.5 ms | Executes parallel dense + sparse leg search |
| **Candidate Union** | Set Union Pool | ~0.6 ms | Merges candidates into top-20 pool |
| **Neural Re-ranking** | `bge-reranker-base` | ~37.0 ms | Fast cross-encoder scoring on candidate snippets |
| **Total Query Latency**| **End-to-End** | **~74.1 ms** | **Well within the <300 ms SLA** |

---

### Reproducing Benchmark Numbers

You can regenerate and verify these metric files locally:

```bash
# 1. Run latency test suite (generates results/phase1_latency.json and phase2_latency.json)
python scripts/benchmark_latency.py

# 2. Run retrieval quality evaluation (generates results/ragas_hybrid.json and phase2_ragas.json)
python eval/eval_ragas.py
---

## Getting Started

### 1. Prerequisites

* **Python 3.10+** (Python 3.11 recommended)
* **Docker & Docker Compose** (for running Qdrant)
* **Git**

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd hydra-rag
```

### 3. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (Command Prompt):
# .venv\Scripts\activate.bat

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env   # Or create it manually
```

Populate it with your credentials:

```ini
# Optional: Required only if using managed cloud Qdrant
QDRANT_API_KEY=

# Required if running RAGAS evaluation with OpenAI/Gemini/Anthropic LLMs
OPENAI_API_KEY=your_llm_api_key_here
```

Review `config.yaml` to ensure Qdrant endpoints, embedding models, and vector collection parameters match your target setup.

---

## Running the Platform

### Step 1: Start the Vector Database (Qdrant)

Launch the Qdrant vector engine using Docker Compose:

```bash
docker compose up -d
```

Verify that the Qdrant container is up and running:

* **Web UI / Dashboard:** `http://localhost:6333/dashboard`
* **Healthcheck:** `curl http://localhost:6333/readyz`

---

### Step 2: Ingest and Index the Corpus

Run the ingestion pipeline to parse `data/passages.jsonl`, generate dense and sparse representations, and upload them to Qdrant:

```bash
# Prepare and clean raw passage data
python ingest/prepare_data.py

# Generate embeddings and upload to Qdrant collection
python ingest/qdrant_ingest.py
```

To verify that the records were correctly populated in Qdrant:

```bash
python scripts/smoke_test.py
```

---

### Step 3: Run Benchmarks and Evaluation (Optional)

**Latency profiling (p50 / p95 / p99).** Executes query runs against dense and hybrid retrieval tracks and writes telemetry logs to `results/`:

```bash
python scripts/benchmark_latency.py
```

**RAGAS quality evaluation.** Evaluates retrieval accuracy, Context Precision, and Context Recall using `data/eval_set.json`:

```bash
python eval/eval_ragas.py
```

---

### Step 4: Launch the Streamlit Dashboard

Run the UI to test interactive queries, explore passage categories, and view side-by-side performance scorecards:

```bash
python -m streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## Retrieval Modes Explained

| Mode | Mechanism | Strength | Latency Target |
| --- | --- | --- | --- |
| **Dense (Phase 1)** | Cosine similarity over BGE dense vector representations. | Captures semantic context and conceptual meaning. | `< 50 ms` |
| **Sparse (BM25)** | Token-level lexical frequency via Qdrant Sparse vectors. | Exact keyword, entity, product, and code matches. | `< 30 ms` |
| **Hybrid (Phase 2)** | Union of candidates from dense + sparse legs, re-ranked via Cross-Encoder. | Maximizes precision and recall; resolves keyword mismatch. | `< 150 ms` |

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'retrieval'`:** Always run commands from the repository root, or explicitly register the workspace root in your environment:

  ```bash
  export PYTHONPATH="${PYTHONPATH}:$(pwd)"
  ```

* **Qdrant Connection Refused:** Ensure the Docker daemon is active and run `docker compose ps` to verify container status on port `6333`.

* **CPU Latency Optimization:** FastEmbed uses ONNX Runtime. Multi-threading is pinned via OpenMP flags in `retrieval/retriever.py` to prevent CPU oversubscription:

  ```python
  import os
  os.environ["OMP_NUM_THREADS"] = "4"
  os.environ["MKL_NUM_THREADS"] = "4"
  ```
