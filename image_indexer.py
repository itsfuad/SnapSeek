from pathlib import Path
from typing import List, Dict, Set, Optional
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
from qdrant_client.http.models import PointStruct
import uuid
from qdrant_singleton import QdrantClientSingleton
from fastapi import WebSocket
from enum import Enum

class IndexingStatus(Enum):
    IDLE = "idle"
    INDEXING = "indexing"
    MONITORING = "monitoring"

class ImageIndexer:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize CLIP model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Qdrant client
        self.collection_name = "images"
        self.qdrant = QdrantClientSingleton.get_instance()
        self.init_collection()
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache of indexed paths
        self.indexed_paths = set()
        
        # Status tracking
        self.status = IndexingStatus.IDLE
        self.current_file: Optional[str] = None
        self.total_files = 0
        self.processed_files = 0
        self.websocket_connections: Set[WebSocket] = set()
    
    def init_collection(self):
        """Initialize Qdrant collection for storing image vectors"""
        try:
            QdrantClientSingleton.initialize_collection(self.collection_name)
            # Load existing indexed paths
            self._load_indexed_paths()
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
    
    def _load_indexed_paths(self):
        """Load the set of already indexed paths from Qdrant"""
        try:
            response = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            self.indexed_paths = {point.payload["path"] for point in response[0]}
        except Exception as e:
            print(f"Error loading indexed paths: {e}")
            self.indexed_paths = set()
    
    async def broadcast_status(self):
        """Broadcast current status to all connected WebSocket clients"""
        status_data = {
            "status": self.status.value,
            "current_file": self.current_file,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "progress_percentage": round((self.processed_files / self.total_files * 100) if self.total_files > 0 else 0, 2)
        }
        
        for connection in self.websocket_connections:
            try:
                await connection.send_json(status_data)
            except Exception as e:
                print(f"Error broadcasting to WebSocket: {e}")
                self.websocket_connections.remove(connection)
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a new WebSocket connection"""
        await websocket.accept()
        self.websocket_connections.add(websocket)
        await self.broadcast_status()
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.websocket_connections.remove(websocket)
    
    async def index_existing_images(self):
        """Index all existing images in the data directory"""
        print("Starting to index existing images...")
        self.status = IndexingStatus.INDEXING
        self.processed_files = 0
        
        # Use rglob for recursive directory scanning
        image_files = [f for f in self.data_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
        self.total_files = len(image_files)
        print(f"Found {self.total_files} images to index")
        
        for i, image_file in enumerate(image_files, 1):
            if str(image_file) not in self.indexed_paths:
                self.current_file = str(image_file)
                print(f"Indexing image {i}/{self.total_files}: {image_file.name}")
                await self.index_image(image_file)
                self.processed_files = i
                await self.broadcast_status()
            else:
                print(f"Skipping already indexed image {i}/{self.total_files}: {image_file.name}")
                self.processed_files = i
                await self.broadcast_status()
        
        self.status = IndexingStatus.MONITORING
        self.current_file = None
        await self.broadcast_status()
        print("Finished indexing all images")
    
    def start_monitoring(self):
        """Start monitoring the data directory for changes"""
        event_handler = ImageEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.data_dir), recursive=True)
        observer.start()
        self.status = IndexingStatus.MONITORING
        print("Started monitoring data directory for changes")
    
    async def index_image(self, image_path: Path):
        """Index a single image"""
        try:
            print(f"Indexing image: {image_path}")
            self.current_file = str(image_path)
            await self.broadcast_status()
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embedding = image_features.cpu().numpy().flatten()
            
            # Verify embedding is valid
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                print(f"Warning: Invalid embedding generated for {image_path}")
                return
            
            # Store in Qdrant
            point_id = str(uuid.uuid4())
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "path": str(image_path),  # Relative path
                            "absolute_path": str(image_path.absolute())  # Absolute path
                        }
                    )
                ]
            )
            self.indexed_paths.add(str(image_path))  # Add to cache
            print(f"Stored embedding in Qdrant for {image_path}")
            
        except Exception as e:
            print(f"Error indexing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.current_file = None
            await self.broadcast_status()
    
    async def get_all_images(self) -> List[Dict]:
        """Get all indexed images"""
        try:
            # Get all points from the collection
            response = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {
                    "path": point.payload["path"],  # Relative path
                    "absolute_path": point.payload["absolute_path"]  # Absolute path
                }
                for point in response[0]
            ]
        except Exception as e:
            print(f"Error getting images: {e}")
            return []

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: ImageIndexer):
        self.indexer = indexer
    
    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(self.indexer.index_image(Path(event.src_path))) 