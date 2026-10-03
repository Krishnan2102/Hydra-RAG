import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from fusion.rank_fusion import reciprocal_rank_fusion, weighted_linear_fusion

CONFIG_PATH = Path("config.yaml")


class HydraRetriever:
    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        q_cfg = self.cfg["qdrant"]
        r_cfg = self.cfg["retrieval"]
        f_cfg = self.cfg.get("fusion", {})

        self.collection_name = q_cfg["collection_name"]
        self.dense_name = q_cfg["dense_vector"]
        self.sparse_name = q_cfg["sparse_vector"]
        self.query_prefix = r_cfg.get("query_prefix", "")
        self.top_k = r_cfg.get("top_k", 5)
        self.candidates_per_leg = r_cfg.get("candidates_per_leg", 50)

        # Fusion settings
        self.fusion_method = f_cfg.get("method", "rrf")
        self.rrf_k = f_cfg.get("rrf_k", 60)
        self.fusion_weights = f_cfg.get("weights", {"dense": 1.0, "sparse": 1.0})

        # Clients and models (all local)
        self.client = QdrantClient(host=q_cfg["host"], port=q_cfg["port"], timeout=30)
        self.dense_model = SentenceTransformer(r_cfg["model_name"], device="cpu")
        self.dense_model.max_seq_length = r_cfg.get("max_seq_length", 256)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filters:
            return None
        must_conditions = []
        for key, value in filters.items():
            if value is not None and value != "" and value != "All":
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        return models.Filter(must=must_conditions) if must_conditions else None

    def search_dense(
        self,
        query: str,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        limit = limit or self.top_k
        t_start = time.perf_counter()

        prefixed_query = f"{self.query_prefix}{query}"
        dense_vec = self.dense_model.encode(prefixed_query, normalize_embeddings=True).tolist()
        t_embed = time.perf_counter()

        qdrant_filter = self._build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using=self.dense_name,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        t_done = time.perf_counter()

        hits = []
        for hit in response.points:
            payload = hit.payload or {}
            hits.append({
                "id": hit.id,
                "score": float(hit.score),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "category": payload.get("category", "unknown"),
                "sub_category": payload.get("sub_category", "unknown")
            })

        return {
            "query": query,
            "mode": "dense",
            "hits": hits,
            "timings_ms": {
                "embed_ms": round((t_embed - t_start) * 1000, 2),
                "search_ms": round((t_done - t_embed) * 1000, 2),
                "total_ms": round((t_done - t_start) * 1000, 2)
            }
        }

    def search_sparse(
        self,
        query: str,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        limit = limit or self.top_k
        t_start = time.perf_counter()

        # BM25 sparse vector generation
        sparse_vec = list(self.sparse_model.embed([query]))[0]
        qdrant_sparse = models.SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist()
        )
        t_embed = time.perf_counter()

        qdrant_filter = self._build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=qdrant_sparse,
            using=self.sparse_name,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        t_done = time.perf_counter()

        hits = []
        for hit in response.points:
            payload = hit.payload or {}
            hits.append({
                "id": hit.id,
                "score": float(hit.score),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "category": payload.get("category", "unknown"),
                "sub_category": payload.get("sub_category", "unknown")
            })

        return {
            "query": query,
            "mode": "sparse",
            "hits": hits,
            "timings_ms": {
                "embed_ms": round((t_embed - t_start) * 1000, 2),
                "search_ms": round((t_done - t_embed) * 1000, 2),
                "total_ms": round((t_done - t_start) * 1000, 2)
            }
        }

    def search_hybrid(
        self,
        query: str,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        limit = limit or self.top_k
        candidates = self.candidates_per_leg
        t_start = time.perf_counter()

        # Execute both legs with candidates pool
        dense_res = self.search_dense(query, limit=candidates, filters=filters)
        sparse_res = self.search_sparse(query, limit=candidates, filters=filters)
        t_legs_done = time.perf_counter()

        ranked_lists = {
            "dense": dense_res["hits"],
            "sparse": sparse_res["hits"]
        }

        # Apply configured fusion algorithm
        if self.fusion_method == "linear":
            fused_hits = weighted_linear_fusion(ranked_lists, weights=self.fusion_weights)
        else:
            fused_hits = reciprocal_rank_fusion(ranked_lists, weights=self.fusion_weights, k=self.rrf_k)

        top_fused = fused_hits[:limit]
        t_done = time.perf_counter()

        return {
            "query": query,
            "mode": "hybrid",
            "fusion_method": self.fusion_method,
            "hits": top_fused,
            "timings_ms": {
                "dense_leg_ms": dense_res["timings_ms"]["total_ms"],
                "sparse_leg_ms": sparse_res["timings_ms"]["total_ms"],
                "fusion_ms": round((t_done - t_legs_done) * 1000, 2),
                "total_ms": round((t_done - t_start) * 1000, 2)
            }
        }