import json
from retrieval.retriever import HydraRetriever


def main():
    print("Initializing retriever...")
    retriever = HydraRetriever()

    test_query = "What are the common symptoms of high blood pressure?"
    print(f"\nRunning dense search for query: '{test_query}'")

    # 1. Unfiltered dense search
    res = retriever.search_dense(test_query, limit=3)
    print(f"Total time: {res['timings_ms']['total_ms']} ms "
          f"(Embed: {res['timings_ms']['embed_ms']} ms | Qdrant: {res['timings_ms']['search_ms']} ms)")
    print(f"Top hit ID: {res['hits'][0]['id']} | Score: {res['hits'][0]['score']:.4f}")
    print(f"Top hit text: {res['hits'][0]['text'][:120]}...")
    print(f"Category: {res['hits'][0]['category']} | Sub-category: {res['hits'][0]['sub_category']}")

    # 2. Filtered dense search (pre-retrieval)
    target_category = res['hits'][0]['category']
    print(f"\nRunning filtered search with category='{target_category}'...")
    res_filtered = retriever.search_dense(test_query, limit=3, filters={"category": target_category})
    print(f"Filtered time: {res_filtered['timings_ms']['total_ms']} ms")
    for hit in res_filtered["hits"]:
        assert hit["category"] == target_category, "Filter leak detected!"
    print("Pre-retrieval metadata filter verified successfully.")


if __name__ == "__main__":
    main()