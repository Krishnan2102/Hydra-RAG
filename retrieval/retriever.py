import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("config.yaml")


class HydraRetriever:
    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        q_cfg = self.cfg["qdrant"]
        r_cfg = self.cfg["retrieval"]

        self.collection_name = q_cfg["collection_name"]
        self.dense_name = q_cfg["dense_vector"]
        self.sparse_name = q_cfg["sparse_vector"]
        self.query_prefix = r_cfg.get("query_prefix", "")
        self.top_k = r_cfg.get("top_k", 5)

        # Initialize Qdrant Client
        self.client = QdrantClient(host=q_cfg["host"], port=q_cfg["port"], timeout=30)

        # Initialize Embedding Models
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

        # 1. Dense query vector computation with instruction prefix
        prefixed_query = f"{self.query_prefix}{query}"
        dense_vec = self.dense_model.encode(
            prefixed_query,
            normalize_embeddings=True
        ).tolist()
        t_embed = time.perf_counter()

        # 2. Vector DB search with native pre-filtering using query_points API
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