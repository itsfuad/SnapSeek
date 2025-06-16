import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from pathlib import Path
import shutil
import os
from unittest.mock import patch

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in os.sys.path:
    os.sys.path.insert(0, project_root)

from app import app, indexer  # Import after path modification

@pytest.fixture(scope="function")
def setup_test_image_folder():
    base_test_dir = Path("test_temp_images")
    indexed_folder = base_test_dir / "indexed_folder"
    indexed_folder.mkdir(parents=True, exist_ok=True)

    test_image_name = "test_image.jpg"
    test_image_path = indexed_folder / test_image_name

    # Create a dummy JPEG image
    img = Image.new('RGB', (600, 400), color = 'red')
    img.save(test_image_path, "JPEG")

    # Create a dummy text file
    test_text_name = "test.txt"
    test_text_path = indexed_folder / test_text_name
    with open(test_text_path, "w") as f:
        f.write("This is a test file.")

    # Mock get_folder_info to simulate an indexed folder
    mock_folder_info = {
        "path": str(indexed_folder.absolute()),
        "collection_name": "test_collection",
        "is_valid": True
    }

    with patch.object(indexer.folder_manager, 'get_folder_info', return_value=mock_folder_info) as mock_get_info:
        yield {
            "base_dir": base_test_dir,
            "indexed_folder_path": indexed_folder,
            "image_name": test_image_name,
            "text_name": test_text_name,
            "mock_get_info": mock_get_info
        }

    # Teardown: remove the temporary directory
    if base_test_dir.exists():
        shutil.rmtree(base_test_dir)

def test_serve_thumbnail_success(client: TestClient, setup_test_image_folder):
    folder_path = setup_test_image_folder["indexed_folder_path"]
    image_name = setup_test_image_folder["image_name"]

    response = client.get(f"/thumbnail/{folder_path}/{image_name}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "max-age=3600" in response.headers["cache-control"] # Check if max-age is present

    img = Image.open(io.BytesIO(response.content))
    assert img is not None
    assert img.width <= 200
    assert img.height <= 200
    assert img.format == "JPEG"

    # Ensure the mock was called for the correct folder path
    setup_test_image_folder["mock_get_info"].assert_called_with(str(folder_path.absolute()))

def test_serve_thumbnail_file_not_found(client: TestClient, setup_test_image_folder):
    folder_path = setup_test_image_folder["indexed_folder_path"]

    response = client.get(f"/thumbnail/{folder_path}/non_existent_image.jpg")

    assert response.status_code == 404
    # Ensure the mock was called for the correct folder path
    setup_test_image_folder["mock_get_info"].assert_called_with(str(folder_path.absolute()))


def test_serve_thumbnail_folder_not_found(client: TestClient):
    unmanaged_folder = "unmanaged_folder"
    image_name = "some_image.jpg"

    # Mock get_folder_info to simulate the folder is not managed/indexed
    with patch.object(indexer.folder_manager, 'get_folder_info', return_value=None) as mock_get_info:
        response = client.get(f"/thumbnail/{unmanaged_folder}/{image_name}")

    assert response.status_code == 404
    mock_get_info.assert_called_with(unmanaged_folder)


def test_serve_thumbnail_invalid_file_type(client: TestClient, setup_test_image_folder):
    folder_path = setup_test_image_folder["indexed_folder_path"]
    text_name = setup_test_image_folder["text_name"]

    response = client.get(f"/thumbnail/{folder_path}/{text_name}")

    assert response.status_code == 400
    # Ensure the mock was called for the correct folder path
    setup_test_image_folder["mock_get_info"].assert_called_with(str(folder_path.absolute()))
