import time
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load config once when module is imported
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize clients globally so the model stays warm in memory
client = QdrantClient(host=config["qdrant"]["host"], port=config["qdrant"]["port"])
model = SentenceTransformer(config["retrieval"]["model_name"])
COLLECTION_NAME = config["qdrant"]["collection_name"]

def search(query: str, mode: str = "dense", top_k: int = 5, filters: dict = None) -> list[dict]:
    """
    Search Qdrant for the closest passages.
    mode and filters are placeholders for Phase 2.
    """
    start_total = time.perf_counter()
    
    # 1. Embed the query with the BGE prefix
    start_embed = time.perf_counter()
    bge_prefix = "Represent this sentence for searching relevant passages: "
    query_vector = model.encode(bge_prefix + query, normalize_embeddings=True).tolist()
    embed_time_ms = (time.perf_counter() - start_embed) * 1000

    # 2. Execute Approximate Nearest Neighbor Search
    start_search = time.perf_counter()
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=top_k
    ).points
    search_time_ms = (time.perf_counter() - start_search) * 1000
    
    # 3. Format the results
    results = []
    for point in search_result:
        results.append({
            "id": point.id,
            "text": point.payload["text"],
            "score": point.score,
            "source": point.payload.get("source", "unknown"),
            "category": point.payload.get("category", "placeholder"),
            "timings": {
                "embed_ms": round(embed_time_ms, 2),
                "search_ms": round(search_time_ms, 2),
                "total_ms": round((time.perf_counter() - start_total) * 1000, 2)
            }
        })
        
    return results

if __name__ == "__main__":
    # Quick test if you run this file directly
    res = search("What causes tides?")
    print(f"Top result score: {res[0]['score']}")
    print(f"Text: {res[0]['text'][:100]}...")
    print(f"Timings: {res[0]['timings']}")