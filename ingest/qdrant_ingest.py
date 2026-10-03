import json
import time
from pathlib import Path
import numpy as np
import yaml
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

CONFIG_PATH = Path("config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    paths = config["paths"]
    qdrant_cfg = config["qdrant"]

    passages_path = Path(paths["passages"])
    embeddings_path = Path(paths["embeddings"])
    categories_path = Path(paths["categories"])

    print("--- 1. Loading Ingestion Assets ---")
    print(f"Loading dense embeddings from {embeddings_path}...")
    dense_embeddings = np.load(embeddings_path)
    total_points = len(dense_embeddings)

    print(f"Loading categories from {categories_path}...")
    with open(categories_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    print(f"Loading passages from {passages_path}...")
    passages = []
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))

    assert len(passages) == total_points == len(categories), (
        f"Length mismatch: {len(passages)} passages vs "
        f"{total_points} embeddings vs {len(categories)} categories"
    )
    print(f"Verified alignment: {total_points} items.")

    print("\n--- 2. Initializing Qdrant Collection ---")
    client = QdrantClient(host=qdrant_cfg["host"], port=qdrant_cfg["port"], timeout=60)
    collection_name = qdrant_cfg["collection_name"]
    dense_name = qdrant_cfg["dense_vector"]
    sparse_name = qdrant_cfg["sparse_vector"]

    # Recreate collection to guarantee a fresh, idempotent state
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' exists. Recreating...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_name: models.VectorParams(
                size=dense_embeddings.shape[1],
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            sparse_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )
    print(f"Collection '{collection_name}' created with dense & sparse vectors.")

    # Create payload indexes for pre-retrieval filtering
    print("Creating keyword payload indexes for filtering...")
    for field_name in ["category", "sub_category", "source"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    print("Payload indexes ready.")

    print("\n--- 3. Computing Sparse Vectors & Ingesting Points ---")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    batch_size = 500
    t_start = time.perf_counter()

    for i in range(0, total_points, batch_size):
        b_end = min(i + batch_size, total_points)
        batch_passages = passages[i:b_end]
        batch_dense = dense_embeddings[i:b_end]
        batch_cats = categories[i:b_end]

        # Compute BM25 sparse vectors via FastEmbed
        batch_texts = [p["text"] for p in batch_passages]
        sparse_gen = list(sparse_model.embed(batch_texts))

        points = []
        for j, (p_info, d_vec, s_vec, cat_info) in enumerate(
            zip(batch_passages, batch_dense, sparse_gen, batch_cats)
        ):
            point_id = int(p_info["id"])
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        dense_name: d_vec.tolist(),
                        sparse_name: models.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist()
                        )
                    },
                    payload={
                        "passage_id": point_id,
                        "text": p_info["text"],
                        "source": p_info.get("source", "unknown"),
                        "category": cat_info["category"],
                        "sub_category": cat_info["sub_category"]
                    }
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

        elapsed = time.perf_counter() - t_start
        rate = b_end / elapsed
        eta_min = (total_points - b_end) / rate / 60 if rate > 0 else 0
        if (i // batch_size) % 10 == 0 or b_end == total_points:
            print(f"Ingested {b_end}/{total_points} points | Elapsed: {elapsed/60:.1f}m | ETA: {eta_min:.1f}m")

    total_time = time.perf_counter() - t_start
    print(f"\nIngestion finished successfully in {total_time/60:.2f} minutes.")
    
    info = client.get_collection(collection_name)
    print(f"Collection status: {info.status} | Total points: {info.points_count}")


if __name__ == "__main__":
    main()