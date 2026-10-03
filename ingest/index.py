import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient, models
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    # Allow passing a sample embeddings file if you want to test before the full run
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")
    args = parser.parse_args()

    config = load_config()
    collection_name = config["qdrant"]["collection_name"]
    embeddings_path = Path(args.embeddings if args.embeddings else config["paths"]["embeddings"])
    passages_path = Path(config["paths"]["passages"])

    print(f"Connecting to Qdrant at {config['qdrant']['host']}:{config['qdrant']['port']}...")
    client = QdrantClient(host=config["qdrant"]["host"], port=config["qdrant"]["port"])

    # Re-runnable: Delete if it already exists
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' exists. Recreating...")
        client.delete_collection(collection_name)

    print("Creating collection with 'dense' and 'sparse' named vectors...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    print("Creating keyword payload indexes for 'category' and 'source'...")
    client.create_payload_index(collection_name, "category", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "source", models.PayloadSchemaType.KEYWORD)

    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    
    print("Uploading points to Qdrant...")
    batch_size = 500
    points = []
    
    with open(passages_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=embeddings.shape[0])):
            # Stop if we reach the end of the loaded embeddings (useful for sample runs)
            if i >= embeddings.shape[0]:
                break
                
            record = json.loads(line)
            
            payload = {
                "passage_id": record["id"],
                "text": record["text"],
                "source": record.get("source", "unknown"),
                "category": "placeholder" # Will be updated in Phase 2
            }
            
            points.append(
                models.PointStruct(
                    id=record["id"],
                    vector={"dense": embeddings[i].tolist()},
                    payload=payload
                )
            )
            
            if len(points) >= batch_size:
                client.upload_points(collection_name=collection_name, points=points, wait=False)
                points = []
                
        if points:
            client.upload_points(collection_name=collection_name, points=points, wait=True)

    info = client.get_collection(collection_name)
    print(f"\nIngestion complete! Total points in '{collection_name}': {info.points_count}")

    print("Running a test retrieval...")
    test_res = client.query_points(
        collection_name=collection_name,
        query=embeddings[0].tolist(),
        using="dense",
        limit=1
    )
    print(f"Test match text: {test_res.points[0].payload['text'][:100]}...")

if __name__ == "__main__":
    main()  