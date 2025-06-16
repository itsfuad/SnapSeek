import os
import sys

import pytest
from fastapi.testclient import TestClient
from app import app # Assuming your FastAPI app instance is named 'app' in 'app.py'

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c