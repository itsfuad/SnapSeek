from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.http.models import Distance, VectorParams

class QdrantClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BaseQdrantClient(path="./qdrant_data")
        return cls._instance

    @classmethod
    def initialize_collection(cls, collection_name: str):
        client = cls.get_instance()
        # Create collection if it doesn't exist
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print(f"Initialized Qdrant collection: {collection_name}") 