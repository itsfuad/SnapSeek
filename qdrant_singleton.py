from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path

CURRENT_SCHEMA_VERSION = "1.0"  # Increment this when schema changes
VECTOR_SIZE = 512  # CLIP embedding size

class QdrantClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Use persistent storage in the data directory
            storage_path = Path("./qdrant_data")
            storage_path.mkdir(exist_ok=True)
            cls._instance = QdrantClient(path=str(storage_path))
        return cls._instance

    @classmethod
    def initialize_collection(cls, collection_name: str):
        client = cls.get_instance()
        
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(collection.name == collection_name for collection in collections)
        
        if not exists:
            # Create new collection with current schema version
            cls._create_collection(client, collection_name)
        else:
            # Check schema version and update if necessary
            cls._check_and_update_schema(client, collection_name)
    
    @classmethod
    def _create_collection(cls, client: QdrantClient, collection_name: str):
        """Create a new collection with the current schema version"""
        # First create the collection with basic config
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            ),
            on_disk_payload=True,  # Store vectors on disk
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0  # Index immediately
            )
        )
        
        # Then create payload indexes for efficient searching
        client.create_payload_index(
            collection_name=collection_name,
            field_name="path",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="absolute_path",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="schema_version",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="indexed_at",
            field_schema=models.PayloadSchemaType.INTEGER
        )
        
        print(f"Created collection {collection_name} with schema version {CURRENT_SCHEMA_VERSION}")
    
    @classmethod
    def _check_and_update_schema(cls, client: QdrantClient, collection_name: str):
        """Check collection schema version and update if necessary"""
        try:
            # Get a sample point to check schema version
            sample = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True
            )[0]
            
            if not sample:
                print(f"Collection {collection_name} is empty")
                return
            
            # Check schema version of existing data
            point_version = sample[0].payload.get("schema_version", "0.0")
            if point_version != CURRENT_SCHEMA_VERSION:
                print(f"Schema version mismatch: {point_version} != {CURRENT_SCHEMA_VERSION}")
                print(f"Collection {collection_name} needs to be recreated")
                
                # Recreate collection with new schema
                client.delete_collection(collection_name=collection_name)
                cls._create_collection(client, collection_name)
            else:
                print(f"Collection {collection_name} schema is up to date (version {CURRENT_SCHEMA_VERSION})")
        except Exception as e:
            print(f"Error checking schema: {e}")
            cls._create_collection(client, collection_name) 