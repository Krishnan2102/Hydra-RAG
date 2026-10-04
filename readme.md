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

The retrieval pipeline was benchmarked across the golden evaluation corpus to compare **Phase 1 (Dense Baseline)** against **Phase 2 (Hybrid Retrieval + Neural Re-ranking)**.

---

### 1. Hackathon Target Scorecard

All key hackathon quality thresholds and latency constraints were successfully satisfied:

| Metric | Phase 1 (Dense) | Phase 2 (Hybrid) | Net Change | Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Precision** | 0.863 | **0.884** | +0.021 | above 0.750 | **Met**[cite: 5] |
| **Context Recall** | 1.000 | 1.000 | +0.000 | above 0.700 | **Met**[cite: 5] |
| **p50 Latency** | 29.7 ms | 196.5 ms | +166.8 ms | — | Evaluated[cite: 5] |
| **p95 Latency** | 37.4 ms | **254.3 ms** | +216.9 ms | under 300.0 ms | **Met**[cite: 5] |
| **p99 Latency** | 45.9 ms | 313.9 ms | +268.1 ms | — | Evaluated[cite: 5] |

---

### 2. Key Findings & Architecture Analysis

* **Target Recall Maintained (1.000):** Context recall comfortably exceeded the minimum 0.700 target threshold across both dense baseline and hybrid tracks[cite: 5].
* **Lifted Context Precision (+0.021):** Introducing the cross-encoder re-ranking stage (`BAAI/bge-reranker-base`) increased context precision from **0.863 to 0.884** (exact score: `0.8838`), filtering out semantic noise and positioning relevant passages higher in top-5 candidate lists[cite: 5].
* **Latency SLA Compliance (< 300 ms):** Although the secondary cross-encoder re-ranking step introduces additional neural inference, Phase 2 maintained a **p95 latency of 254.3 ms**, staying within the **< 300.0 ms** hackathon performance requirement[cite: 5].

---

### 3. Reproducing the Evaluation Telemetry

To re-run the benchmark tests and recreate the scorecard locally:

```bash
# 1. Run latency benchmarks (writes to results/phase1_latency.json and phase2_latency.json)
python scripts/benchmark_latency.py

# 2. Run automated RAGAS scoring (writes to results/phase1_ragas.json and phase2_ragas.json)
python eval/eval_ragas.py

# 3. View the live scorecard and interactive plots in the Streamlit UI
python -m streamlit run app/streamlit_app.py

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
