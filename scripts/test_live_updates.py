from ingest.index_service import IndexService
from retrieval.retriever import HydraRetriever


def main():
    service = IndexService()
    retriever = HydraRetriever()

    test_id = 999999
    secret_text = "Hydra RAG introduces high precision dual vector indexing for Adrosonic hackathon 2026."

    print("--- 1. Testing Live Upsert ---")
    res_up = service.upsert_passage(
        passage_id=test_id,
        text=secret_text,
        category="hackathon_entry",
        sub_category="adrosonic_live"
    )
    print(f"Upserted ID {test_id} in {res_up['latency_ms']} ms.")

    print("\n--- 2. Verifying Immediate Retrieval via Hybrid Search ---")
    search_res = retriever.search_hybrid("Adrosonic hackathon 2026 Hydra RAG", limit=3)
    top_hit = search_res["hits"][0]
    print(f"Top Hit ID: {top_hit['id']} | Fused Score: {top_hit['fused_score']}")
    print(f"Text: {top_hit['text']}")
    assert top_hit["id"] == test_id, f"Expected ID {test_id}, got {top_hit['id']}"
    print("Live upsert retrieval verified!")

    print("\n--- 3. Testing Live Delete ---")
    res_del = service.delete_passage(test_id)
    print(f"Deleted ID {test_id} in {res_del['latency_ms']} ms.")

    print("\n--- 4. Verifying Point Is Removed ---")
    check_points = retriever.client.retrieve(
        collection_name=retriever.collection_name,
        ids=[test_id]
    )
    assert len(check_points) == 0, "Point should no longer exist in Qdrant!"
    print("Live deletion verified successfully!")


if __name__ == "__main__":
    main()