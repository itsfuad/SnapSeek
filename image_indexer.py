from pathlib import Path
from typing import List, Dict, Set, Optional
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor, ThreadPoolExecutor
import threading
from qdrant_client.http.models import PointStruct
import uuid
from qdrant_singleton import QdrantClientSingleton, CURRENT_SCHEMA_VERSION
from fastapi import WebSocket
from enum import Enum
import qdrant_client
import time

class IndexingStatus(Enum):
    IDLE = "idle"
    INDEXING = "indexing"
    MONITORING = "monitoring"

class ImageIndexer:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize status tracking
        self.status = IndexingStatus.IDLE
        self.current_file: Optional[str] = None
        self.total_files = 0
        self.processed_files = 0
        self.websocket_connections: Set[WebSocket] = set()
        
        # Thread synchronization
        self.collection_initialized = threading.Event()
        self.model_initialized = threading.Event()
        
        # Initialize Qdrant client and collection
        self.collection_name = "images"
        self.qdrant = QdrantClientSingleton.get_instance()
        self.init_collection()
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache of indexed paths
        self.indexed_paths = set()
        
        # Model initialization flags
        self.model = None
        self.processor = None
        self.device = None
        
        # Start model initialization in a separate thread
        threading.Thread(target=self._initialize_model_thread, daemon=True).start()
    
    def init_collection(self):
        """Initialize Qdrant collection for storing image vectors"""
        try:
            QdrantClientSingleton.initialize_collection(self.collection_name)
            # Load existing indexed paths
            self._load_indexed_paths()
            self.collection_initialized.set()
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
            raise  # Re-raise the exception to prevent further processing
    
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
        
        # Start indexing in a separate thread
        threading.Thread(target=self._index_existing_images_thread, daemon=True).start()
    
    def _index_existing_images_thread(self):
        """Index existing images in a separate thread"""
        asyncio.run(self.index_existing_images())
    
    async def initialize_model(self):
        """Initialize CLIP model and processor in the background"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model_initialized.set()
            print("Model initialization complete")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.status = IndexingStatus.IDLE
            await self.broadcast_status()
    
    def _initialize_model_thread(self):
        """Initialize model in a separate thread"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model_initialized.set()
            print("Model initialization complete")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.status = IndexingStatus.IDLE
            asyncio.run(self.broadcast_status())
    
    async def index_image(self, image_path: Path):
        """Index a single image"""
        try:
            # Wait for both model and collection initialization
            while not self.model_initialized.is_set():
                await asyncio.sleep(0.1)
            
            if not self.collection_initialized.is_set():
                print("Waiting for collection initialization...")
                self.collection_initialized.wait()
            
            # Convert to relative path from data directory
            try:
                relative_path = image_path.relative_to(self.data_dir)
            except ValueError:
                # If path is already relative or can't be made relative to data_dir
                relative_path = image_path
            
            print(f"Indexing image: {relative_path}")
            self.current_file = str(relative_path)
            await self.broadcast_status()
            
            # Check if image already exists in Qdrant with current schema version
            existing_points = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_client.http.models.Filter(
                    must=[
                        qdrant_client.http.models.FieldCondition(
                            key="path",
                            match={"value": str(relative_path)}
                        ),
                        qdrant_client.http.models.FieldCondition(
                            key="schema_version",
                            match={"value": CURRENT_SCHEMA_VERSION}
                        )
                    ]
                ),
                limit=1
            )[0]
            
            # Skip if image exists with current schema version
            if existing_points:
                print(f"Skipping {relative_path} - already indexed with current schema version")
                return
            
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
                print(f"Warning: Invalid embedding generated for {relative_path}")
                return
            
            # Delete any old versions if they exist
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_client.http.models.FilterSelector(
                    filter=qdrant_client.http.models.Filter(
                        must=[
                            qdrant_client.http.models.FieldCondition(
                                key="path",
                                match={"value": str(relative_path)}
                            )
                        ]
                    )
                )
            )
            
            # Store in Qdrant with schema version and timestamp
            point_id = str(uuid.uuid4())
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "path": str(relative_path),  # Relative path from data directory
                            "absolute_path": str(image_path.absolute()),  # Absolute path
                            "schema_version": CURRENT_SCHEMA_VERSION,
                            "indexed_at": int(time.time())
                        }
                    )
                ]
            )
            self.indexed_paths.add(str(relative_path))  # Add to cache using relative path
            print(f"Stored embedding in Qdrant for {relative_path}")
            
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
            
            # Use a dictionary to ensure unique paths
            unique_images = {}
            for point in response[0]:
                path = point.payload["path"]
                if path not in unique_images:
                    unique_images[path] = {
                        "path": path,  # Relative path
                        "absolute_path": point.payload["absolute_path"],  # Absolute path
                        "indexed_at": point.payload.get("indexed_at", 0)  # Include timestamp
                    }
            
            # Convert to list and sort by indexed_at timestamp (newest first)
            images = list(unique_images.values())
            images.sort(key=lambda x: x["indexed_at"], reverse=True)
            
            return images
            
        except Exception as e:
            print(f"Error getting images: {e}")
            import traceback
            traceback.print_exc()
            return []

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: ImageIndexer):
        self.indexer = indexer
    
    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(self.indexer.index_image(Path(event.src_path))) 