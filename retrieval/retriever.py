import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models


def load_config() -> Dict[str, Any]:
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HydraRetriever:
    """
    Two-Stage Hybrid Retriever:
    Stage 1: Multi-channel candidate retrieval (Dense + BM25 Sparse).
    Stage 2: Cross-Encoder neural re-ranking over candidate union pool.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or load_config()

        # Resolve Qdrant parameters
        qdrant_cfg = self.cfg.get("qdrant", {})
        self.collection_name = qdrant_cfg.get("collection_name", "hydra_passages")
        qdrant_url = qdrant_cfg.get("url")
        if not qdrant_url:
            host = qdrant_cfg.get("host", "localhost")
            port = qdrant_cfg.get("port", 6333)
            qdrant_url = f"http://{host}:{port}"

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY", None),
        )

        # Resolve models
        embed_cfg = self.cfg.get("embedding", {})
        retrieval_cfg = self.cfg.get("retrieval", {})
        rerank_cfg = self.cfg.get("rerank", {})

        dense_model = (
            embed_cfg.get("dense_model")
            or retrieval_cfg.get("model_name")
            or "BAAI/bge-small-en-v1.5"
        )
        sparse_model = embed_cfg.get("sparse_model", "Qdrant/bm25")
        reranker_model = (
            rerank_cfg.get("model")
            or embed_cfg.get("reranker_model")
            or "BAAI/bge-reranker-base"
        )

        self.dense_vector_name = qdrant_cfg.get("dense_vector", "dense")
        self.sparse_vector_name = qdrant_cfg.get("sparse_vector", "sparse")

        self.rerank_enabled = rerank_cfg.get("enabled", True)
        self.candidate_pool_size = rerank_cfg.get("pool_size", 20)
        self.top_k = retrieval_cfg.get("top_k", 5)

        # FastEmbed instances
        self.dense_embedder = TextEmbedding(model_name=dense_model)
        self.sparse_embedder = SparseTextEmbedding(model_name=sparse_model)

        if self.rerank_enabled:
            self.reranker = TextCrossEncoder(model_name=reranker_model)
        else:
            self.reranker = None

    def embed_dense(self, text: str) -> List[float]:
        return list(self.dense_embedder.embed([text]))[0].tolist()

    def embed_sparse(self, text: str) -> models.SparseVector:
        sparse_gen = list(self.sparse_embedder.embed([text]))[0]
        return models.SparseVector(
            indices=sparse_gen.indices.tolist(),
            values=sparse_gen.values.tolist(),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Pure Dense Baseline Search
    # -------------------------------------------------------------------------
    def search_dense(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        k = limit or self.top_k

        t_embed_0 = time.perf_counter()
        dense_vec = self.embed_dense(query)
        t_embed = (time.perf_counter() - t_embed_0) * 1000.0

        t_search_0 = time.perf_counter()
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using=self.dense_vector_name,
            limit=k,
        ).points
        t_search = (time.perf_counter() - t_search_0) * 1000.0

        hits = []
        for p in points:
            payload = p.payload or {}
            hits.append(
                {
                    "id": p.id,
                    "score": round(float(p.score), 4),
                    "text": payload.get("text", payload.get("content", "")),
                    "metadata": payload,
                }
            )

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "query": query,
            "hits": hits,
            "timings_ms": {
                "total_ms": round(total_ms, 2),
                "dense_leg_ms": round(total_ms, 2),
                "sparse_leg_ms": 0.0,
                "fusion_ms": 0.0,
                "fuse_ms": 0.0,
                "embed_ms": round(t_embed, 2),
                "search_ms": round(t_search, 2),
                "rerank_ms": 0.0,
            },
        }

    # -------------------------------------------------------------------------
    # Sparse Helper (BM25 Only)
    # -------------------------------------------------------------------------
    def search_sparse(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        k = limit or self.top_k

        t_embed_0 = time.perf_counter()
        sparse_vec = self.embed_sparse(query)
        t_embed = (time.perf_counter() - t_embed_0) * 1000.0

        t_search_0 = time.perf_counter()
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using=self.sparse_vector_name,
            limit=k,
        ).points
        t_search = (time.perf_counter() - t_search_0) * 1000.0

        hits = []
        for p in points:
            payload = p.payload or {}
            hits.append(
                {
                    "id": p.id,
                    "score": round(float(p.score), 4),
                    "text": payload.get("text", payload.get("content", "")),
                    "metadata": payload,
                }
            )

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "query": query,
            "hits": hits,
            "timings_ms": {
                "total_ms": round(total_ms, 2),
                "dense_leg_ms": 0.0,
                "sparse_leg_ms": round(total_ms, 2),
                "fusion_ms": 0.0,
                "fuse_ms": 0.0,
                "embed_ms": round(t_embed, 2),
                "search_ms": round(t_search, 2),
                "rerank_ms": 0.0,
            },
        }

    # -------------------------------------------------------------------------
    # Phase 2: Hybrid Search (Retrieve + Rerank)
    # -------------------------------------------------------------------------
    def search_hybrid(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        k = limit or self.top_k
        pool_size = max(self.candidate_pool_size, k)

        # Stage 1: Candidate Generation (Compact Horizon)
        t_dense_0 = time.perf_counter()
        dense_res = self.search_dense(query, limit=pool_size)
        t_dense_leg = (time.perf_counter() - t_dense_0) * 1000.0

        t_sparse_0 = time.perf_counter()
        sparse_res = self.search_sparse(query, limit=pool_size)
        t_sparse_leg = (time.perf_counter() - t_sparse_0) * 1000.0

        dense_hits = dense_res["hits"]
        sparse_hits = sparse_res["hits"]

        candidate_map: Dict[Any, Dict[str, Any]] = {}
        for h in dense_hits:
            candidate_map[h["id"]] = h
        for h in sparse_hits:
            if h["id"] not in candidate_map:
                candidate_map[h["id"]] = h

        candidates = list(candidate_map.values())
        if not candidates:
            return {
                "query": query,
                "hits": [],
                "timings_ms": {
                    "total_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "dense_leg_ms": round(t_dense_leg, 2),
                    "sparse_leg_ms": round(t_sparse_leg, 2),
                    "fusion_ms": 0.0,
                    "fuse_ms": 0.0,
                    "embed_ms": 0.0,
                    "search_ms": 0.0,
                    "rerank_ms": 0.0,
                },
            }

      # Stage 2: Fast Neural Reranking
        t_rerank_0 = time.perf_counter()
        if self.rerank_enabled and self.reranker:
            # Truncate to first 256 characters to stay within <150ms CPU budget
            documents = [c["text"][:256] for c in candidates]
            scores = list(self.reranker.rerank(query, documents))
            for i, score in enumerate(scores):
                candidates[i]["score"] = round(float(score), 4)
                candidates[i]["metadata"]["rerank_score"] = round(float(score), 4)
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)
        t_rerank = (time.perf_counter() - t_rerank_0) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        embed_ms = dense_res["timings_ms"].get("embed_ms", 0.0) + sparse_res["timings_ms"].get("embed_ms", 0.0)
        search_ms = dense_res["timings_ms"].get("search_ms", 0.0) + sparse_res["timings_ms"].get("search_ms", 0.0)

        return {
            "query": query,
            "hits": candidates[:k],
            "timings_ms": {
                "total_ms": round(total_ms, 2),
                "dense_leg_ms": round(t_dense_leg, 2),
                "sparse_leg_ms": round(t_sparse_leg, 2),
                "fusion_ms": round(t_rerank, 2),
                "fuse_ms": round(t_rerank, 2),
                "embed_ms": round(embed_ms, 2),
                "search_ms": round(search_ms, 2),
                "rerank_ms": round(t_rerank, 2),
            },
        }