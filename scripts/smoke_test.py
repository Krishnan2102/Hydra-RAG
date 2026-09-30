from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
name = "smoke_test"
if client.collection_exists(name):
    client.delete_collection(name)

client.create_collection(
    collection_name=name,
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)
client.create_payload_index(name, field_name="category",
                            field_schema=models.PayloadSchemaType.KEYWORD)
client.upsert(name, points=[models.PointStruct(
    id=1,
    vector={"dense": [0.1] * 384,
            "sparse": models.SparseVector(indices=[1, 5, 9], values=[0.5, 0.3, 0.2])},
    payload={"category": "test", "source": "smoke"},
)])

print(client.retrieve(name, ids=[1], with_payload=True))
hits = client.query_points(name, query=[0.1] * 384, using="dense", limit=1)
print(hits.points)
client.delete_collection(name)
print("OK")
