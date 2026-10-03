# HYDRA-RAG

Vector Database Design for Large-Scale Precision Retrieval in RAG Systems.
Phase 1 implementation: Dense Retrieval Baseline.

## Infrastructure Setup

1. Initialize a Python 3.11 virtual environment and activate it.
2. Install the CPU-optimized PyTorch binary:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Resolve system dependencies:
   `pip install -r requirements.txt`
4. Provision the Qdrant vector database via Docker:
   `docker compose up -d`
5. Configure environment variables by copying `.env.example` to `.env` and inserting your Groq API key.

## Execution Pipeline

Execute the ingestion and indexing modules sequentially:
1. `python ingest/prepare_data.py`
2. `python ingest/embed.py`
3. `python ingest/index.py`

## Application Interface

Initialize the Streamlit frontend to execute queries and monitor per-stage latency telemetry:
`streamlit run app/streamlit_app.py`

## Evaluation

Execute the RAGAS contextual precision evaluation and latency benchmarks:
1. `python eval/run_ragas.py`
2. `python eval/latency.py`