import time
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class IndexService:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        q_cfg = cfg["qdrant"]
        r_cfg = cfg["retrieval"]

        self.collection_name = q_cfg["collection_name"]
        self.dense_name = q_cfg["dense_vector"]
        self.sparse_name = q_cfg["sparse_vector"]
        self.query_prefix = r_cfg.get("query_prefix", "")

        self.client = QdrantClient(host=q_cfg["host"], port=q_cfg["port"], timeout=30)
        self.dense_model = SentenceTransformer(r_cfg["model_name"], device="cpu")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def upsert_passage(
        self,
        passage_id: int,
        text: str,
        category: str = "custom",
        sub_category: str = "custom",
        source: str = "live_update"
    ) -> Dict[str, Any]:
        """Upsert a single passage into Qdrant computing both dense and sparse vectors on the fly."""
        t_start = time.perf_counter()

        # Compute dense embedding
        dense_vec = self.dense_model.encode(text, normalize_embeddings=True).tolist()

        # Compute sparse embedding
        sparse_vec = list(self.sparse_model.embed([text]))[0]

        point = models.PointStruct(
            id=passage_id,
            vector={
                self.dense_name: dense_vec,
                self.sparse_name: models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            },
            payload={
                "passage_id": passage_id,
                "text": text,
                "source": source,
                "category": category,
                "sub_category": sub_category
            }
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True
        )
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        return {"status": "success", "id": passage_id, "latency_ms": elapsed}

    def delete_passage(self, passage_id: int) -> Dict[str, Any]:
        """Delete a passage by ID immediately."""
        t_start = time.perf_counter()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[passage_id]),
            wait=True
        )
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        return {"status": "deleted", "id": passage_id, "latency_ms": elapsed}