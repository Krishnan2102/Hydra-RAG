import json
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")
N_TOP_CLUSTERS = 8
N_SUB_CLUSTERS = 3
RANDOM_STATE = 42
BATCH_SIZE = 2048


def get_top_terms(texts: list[str], top_n: int = 2) -> str:
    """Extract top distinctive keywords from a collection of text snippets."""
    if not texts or all(not t.strip() for t in texts):
        return "general"
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
        scores = np.asarray(tfidf.mean(axis=0)).ravel()
        top_indices = scores.argsort()[-top_n:][::-1]
        feature_names = np.array(vectorizer.get_feature_names_out())
        return "_".join(feature_names[top_indices]).replace(" ", "_")
    except Exception:
        return "general"


def main():
    passages_file = DATA_DIR / "passages.jsonl"
    embeddings_file = DATA_DIR / "embeddings.npy"
    output_file = DATA_DIR / "categories.json"

    if not passages_file.exists() or not embeddings_file.exists():
        raise FileNotFoundError("passages.jsonl or embeddings.npy missing from data/")

    print("Loading precomputed dense embeddings...")
    embeddings = np.load(embeddings_file)
    n_points = len(embeddings)
    print(f"Loaded {n_points} embeddings of shape {embeddings.shape}")

    print("Reading text passages for TF-IDF naming...")
    texts = []
    with open(passages_file, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    assert len(texts) == n_points, f"Count mismatch: {len(texts)} texts vs {n_points} embeddings"

    # Step 1: Top-level clustering
    print(f"\n[1/3] Partitioning into {N_TOP_CLUSTERS} primary categories...")
    t0 = time.perf_counter()
    top_kmeans = MiniBatchKMeans(
        n_clusters=N_TOP_CLUSTERS,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        n_init=3
    )
    top_labels = top_kmeans.fit_predict(embeddings)
    print(f"Top-level clustering completed in {time.perf_counter() - t0:.1f}s")

    # Step 2: Name primary categories using sampled snippets
    print("\n[2/3] Extracting primary category labels...")
    top_cluster_names = {}
    for cluster_id in range(N_TOP_CLUSTERS):
        indices = np.where(top_labels == cluster_id)[0]
        # Sample up to 1000 passages to keep TF-IDF fast
        sample_indices = np.random.choice(indices, size=min(1000, len(indices)), replace=False)
        sample_texts = [texts[idx][:250] for idx in sample_indices]
        label_name = get_top_terms(sample_texts, top_n=2)
        top_cluster_names[cluster_id] = label_name
        print(f"  Primary Cluster {cluster_id} ({len(indices)} docs) -> '{label_name}'")

    # Step 3: Hierarchical sub-clustering within each primary cluster
    print(f"\n[3/3] Deriving nested sub-categories ({N_SUB_CLUSTERS} per primary cluster)...")
    t1 = time.perf_counter()
    metadata_records = [None] * n_points

    for cluster_id in range(N_TOP_CLUSTERS):
        indices = np.where(top_labels == cluster_id)[0]
        sub_embeddings = embeddings[indices]
        primary_name = top_cluster_names[cluster_id]

        sub_kmeans = MiniBatchKMeans(
            n_clusters=N_SUB_CLUSTERS,
            batch_size=min(BATCH_SIZE, len(indices)),
            random_state=RANDOM_STATE,
            n_init=3
        )
        sub_labels = sub_kmeans.fit_predict(sub_embeddings)

        sub_cluster_names = {}
        for sub_id in range(N_SUB_CLUSTERS):
            sub_idx_in_cluster = np.where(sub_labels == sub_id)[0]
            global_indices = indices[sub_idx_in_cluster]
            sample_sub_indices = np.random.choice(
                global_indices, size=min(500, len(global_indices)), replace=False
            )
            sample_texts = [texts[idx][:250] for idx in sample_sub_indices]
            sub_name = get_top_terms(sample_texts, top_n=2)
            sub_cluster_names[sub_id] = sub_name
            print(f"    Sub-cluster {primary_name} -> '{sub_name}' ({len(global_indices)} docs)")

        for local_idx, sub_id in enumerate(sub_labels):
            global_id = indices[local_idx]
            metadata_records[global_id] = {
                "category": primary_name,
                "sub_category": sub_cluster_names[sub_id]
            }

    print(f"Sub-clustering completed in {time.perf_counter() - t1:.1f}s")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f)

    print(f"\nDone! Saved {len(metadata_records)} metadata records to {output_file}")


if __name__ == "__main__":
    main()