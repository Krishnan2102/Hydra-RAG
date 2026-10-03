import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models


def load_config() -> Dict[str, Any]:
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HydraRetriever:
    """
    Dual-vector retriever combining Dense (BGE-small) and Sparse (BM25)
    embeddings stored in Qdrant with Calibrated Linear Score Fusion.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or load_config()

        qdrant_cfg = self.cfg.get("qdrant", {})
        self.collection_name = qdrant_cfg.get("collection_name", "hydra_rag")
        self.client = QdrantClient(
            url=qdrant_cfg.get("url", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", None),
        )

        embed_cfg = self.cfg.get("embedding", {})
        dense_model = embed_cfg.get("dense_model", "BAAI/bge-small-en-v1.5")
        sparse_model = embed_cfg.get("sparse_model", "Qdrant/bm25")

        # Initialize FastEmbed dense and sparse models
        self.dense_embedder = TextEmbedding(model_name=dense_model)
        self.sparse_embedder = SparseTextEmbedding(model_name=sparse_model)

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
    def search_dense(self, query: str, limit: int = 5) -> Dict[str, Any]:
        dense_vec = self.embed_dense(query)

        points = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
            limit=limit,
        ).points

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

        return {"query": query, "hits": hits}

    # -------------------------------------------------------------------------
    # Sparse Helper (BM25 Only)
    # -------------------------------------------------------------------------
    def search_sparse(self, query: str, limit: int = 5) -> Dict[str, Any]:
        sparse_vec = self.embed_sparse(query)

        points = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vec,
            using="sparse",
            limit=limit,
        ).points

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

        return {"query": query, "hits": hits}

    # -------------------------------------------------------------------------
    # Phase 2: Hybrid Search (Calibrated Linear Fusion)
    # -------------------------------------------------------------------------
    def search_hybrid(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Anchors strictly on the top-5 dense candidates to lock recall at 1.0000.
        Applies a normalized, bounded BM25 raw-score boost (beta = 0.035) to 
        break dense semantic ties without displacing high-confidence dense matches.
        """
        # 1. Candidate anchor: Top 5 dense hits (protects recall floor)
        dense_hits = self.search_dense(query, limit=limit)["hits"]

        # 2. Expand sparse horizon to capture BM25 scores for all dense candidates
        sparse_hits = self.search_sparse(query, limit=25)["hits"]
        
        # Map BM25 document IDs to their raw scores
        sparse_score_map = {hit["id"]: hit["score"] for hit in sparse_hits}
        
        # Find max sparse score for [0, 1] normalization (fallback to 1.0 to avoid division by zero)
        max_sparse_score = max(sparse_score_map.values()) if sparse_score_map else 1.0

        # 3. Calibrated Score Fusion
        beta = 0.035  # Maximum permitted lexical boost
        
        scored_hits = []
        for hit in dense_hits:
            doc_id = hit["id"]
            dense_score = hit["score"]
            
            # Fetch BM25 score (0.0 if it didn't make the top 25)
            sparse_score = sparse_score_map.get(doc_id, 0.0)
            
            # Calculate normalized boost
            normalized_sparse = sparse_score / max_sparse_score
            lexical_boost = beta * normalized_sparse
            
            # Final fused score
            final_score = round(dense_score + lexical_boost, 4)
            
            # Update the hit with the new score and retain sub-scores for transparency in UI
            hit["score"] = final_score
            hit["metadata"]["dense_score"] = dense_score
            hit["metadata"]["sparse_boost"] = round(lexical_boost, 4)
            
            scored_hits.append(hit)

        # 4. Sort descending by the new fused score
        scored_hits.sort(key=lambda x: x["score"], reverse=True)
        
        return {"query": query, "hits": scored_hits[:limit]}