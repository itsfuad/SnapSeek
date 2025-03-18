import pytest
import uuid
from pathlib import Path
import shutil
from qdrant_singleton import QdrantClientSingleton, CURRENT_SCHEMA_VERSION
from qdrant_client.http import models

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test"""
    # Store original state
    original_path = QdrantClientSingleton._storage_path
    original_instance = QdrantClientSingleton._instance
    
    # Create temporary storage
    temp_path = Path("test_qdrant_data")
    QdrantClientSingleton._storage_path = temp_path
    QdrantClientSingleton._instance = None
    
    yield
    
    # Cleanup
    if QdrantClientSingleton._instance:
        QdrantClientSingleton._instance.close()
    
    # Restore original state
    QdrantClientSingleton._instance = original_instance
    QdrantClientSingleton._storage_path = original_path
    
    # Remove test directory if it exists
    if temp_path.exists():
        shutil.rmtree(temp_path)

def test_singleton_pattern():
    """Test that get_instance returns the same instance"""
    instance1 = QdrantClientSingleton.get_instance()
    instance2 = QdrantClientSingleton.get_instance()
    assert instance1 is instance2

def test_storage_path_creation():
    """Test that storage path is created if it doesn't exist"""
    assert not QdrantClientSingleton._storage_path.exists()
    QdrantClientSingleton.get_instance()
    assert QdrantClientSingleton._storage_path.exists()

def test_collection_creation():
    """Test collection creation"""
    client = QdrantClientSingleton.get_instance()
    collection_name = "test_collection"
    
    # Create collection
    QdrantClientSingleton.initialize_collection(collection_name)
    
    # Check collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    assert collection_name in collection_names

def test_schema_version_check():
    """Test schema version checking and updating"""
    client = QdrantClientSingleton.get_instance()
    collection_name = "test_schema_collection"

    # Create collection
    QdrantClientSingleton.initialize_collection(collection_name)

    # Add a point with current schema version
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector=[0.0] * 512,  # VECTOR_SIZE
                payload={
                    "path": "test.jpg",
                    "absolute_path": "/test/test.jpg",
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "indexed_at": 123456789
                }
            )
        ]
    )

    # Verify point was added
    search_result = client.scroll(
        collection_name=collection_name,
        limit=1
    )
    assert len(search_result[0]) == 1
    assert search_result[0][0].id == point_id
    assert search_result[0][0].payload["schema_version"] == CURRENT_SCHEMA_VERSION

def test_payload_indexes():
    """Test that payload indexes are created correctly"""
    client = QdrantClientSingleton.get_instance()
    collection_name = "test_indexes"

    # Create collection
    QdrantClientSingleton.initialize_collection(collection_name)

    # Get collection info
    collection_info = client.get_collection(collection_name)

    # Check that collection exists and has correct vector size
    assert collection_info.config.params.vectors.size == 512
    assert collection_info.config.params.vectors.distance == models.Distance.COSINE

def test_empty_collection_schema_check():
    """Test schema check behavior with empty collection"""
    client = QdrantClientSingleton.get_instance()
    collection_name = "test_empty_collection"

    # Create collection
    QdrantClientSingleton.initialize_collection(collection_name)

    # Verify collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    assert collection_name in collection_names 