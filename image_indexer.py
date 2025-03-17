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
import threading
from qdrant_client.http.models import PointStruct
import uuid
from qdrant_singleton import QdrantClientSingleton, CURRENT_SCHEMA_VERSION
from fastapi import WebSocket
from enum import Enum
import qdrant_client
import time
from folder_manager import FolderManager

class IndexingStatus(Enum):
    IDLE = "idle"
    INDEXING = "indexing"
    MONITORING = "monitoring"

class ImageIndexer:
    def __init__(self):
        # Initialize folder manager
        self.folder_manager = FolderManager()
        
        # Initialize status tracking
        self.status = IndexingStatus.IDLE
        self.current_file: Optional[str] = None
        self.total_files = 0
        self.processed_files = 0
        self.websocket_connections: Set[WebSocket] = set()
        
        # Thread synchronization
        self.collection_initialized = threading.Event()
        self.model_initialized = threading.Event()
        
        # Initialize Qdrant client
        self.qdrant = QdrantClientSingleton.get_instance()
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache of indexed paths per collection
        self.indexed_paths: Dict[str, Set[str]] = {}
        
        # Model initialization flags
        self.model = None
        self.processor = None
        self.device = None
        
        # Start model initialization in a separate thread
        threading.Thread(target=self._initialize_model_thread, daemon=True).start()
    
    def _load_indexed_paths(self, collection_name: str):
        """Load the set of already indexed paths from a collection"""
        try:
            response = self.qdrant.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            self.indexed_paths[collection_name] = {point.payload["path"] for point in response[0]}
        except Exception as e:
            print(f"Error loading indexed paths for collection {collection_name}: {e}")
            self.indexed_paths[collection_name] = set()
    
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
    
    async def add_folder(self, folder_path: str) -> Dict:
        """Add a new folder to index"""
        folder_info = self.folder_manager.add_folder(folder_path)
        # Start indexing the new folder
        await self.index_folder(folder_path)
        return folder_info
    
    async def remove_folder(self, folder_path: str):
        """Remove a folder from indexing"""
        self.folder_manager.remove_folder(folder_path)
    
    async def index_folder(self, folder_path: str):
        """Index all images in a specific folder"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            return
        
        collection_name = self.folder_manager.get_collection_for_path(folder_path)
        if not collection_name:
            print(f"No collection found for folder: {folder_path}")
            return
        
        print(f"Starting to index folder: {folder_path}")
        self.status = IndexingStatus.INDEXING
        self.processed_files = 0
        
        # Load indexed paths for this collection if not already loaded
        if collection_name not in self.indexed_paths:
            self._load_indexed_paths(collection_name)
        
        # Use rglob for recursive directory scanning
        image_files = [f for f in folder_path.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
        self.total_files = len(image_files)
        print(f"Found {self.total_files} images to index")
        
        for i, image_file in enumerate(image_files, 1):
            relative_path = str(image_file.relative_to(folder_path))
            if relative_path not in self.indexed_paths[collection_name]:
                self.current_file = str(image_file)
                print(f"Indexing image {i}/{self.total_files}: {image_file.name}")
                await self.index_image(image_file, folder_path)
                self.processed_files = i
                await self.broadcast_status()
            else:
                print(f"Skipping already indexed image {i}/{self.total_files}: {image_file.name}")
                self.processed_files = i
                await self.broadcast_status()
        
        # Update last indexed timestamp
        self.folder_manager.update_last_indexed(str(folder_path))
        
        self.status = IndexingStatus.MONITORING
        self.current_file = None
        await self.broadcast_status()
        print("Finished indexing folder")
    
    async def index_image(self, image_path: Path, root_folder: Path):
        """Index a single image"""
        try:
            # Wait for model initialization
            while not self.model_initialized.is_set():
                await asyncio.sleep(0.1)
            
            # Get the collection for this path
            collection_name = self.folder_manager.get_collection_for_path(str(root_folder))
            if not collection_name:
                print(f"No collection found for image: {image_path}")
                return
            
            # Convert to relative path from root folder
            try:
                relative_path = str(image_path.relative_to(root_folder))
            except ValueError:
                print(f"Image {image_path} is not under root folder {root_folder}")
                return
            
            print(f"Indexing image: {relative_path}")
            self.current_file = str(image_path)
            await self.broadcast_status()
            
            # Check if image already exists in Qdrant with current schema version
            existing_points = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_client.http.models.Filter(
                    must=[
                        qdrant_client.http.models.FieldCondition(
                            key="path",
                            match={"value": relative_path}
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
                collection_name=collection_name,
                points_selector=qdrant_client.http.models.FilterSelector(
                    filter=qdrant_client.http.models.Filter(
                        must=[
                            qdrant_client.http.models.FieldCondition(
                                key="path",
                                match={"value": relative_path}
                            )
                        ]
                    )
                )
            )
            
            # Store in Qdrant with schema version and timestamp
            point_id = str(uuid.uuid4())
            self.qdrant.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "path": relative_path,  # Relative path from root folder
                            "absolute_path": str(image_path.absolute()),  # Absolute path
                            "root_folder": str(root_folder.absolute()),  # Store root folder path
                            "schema_version": CURRENT_SCHEMA_VERSION,
                            "indexed_at": int(time.time())
                        }
                    )
                ]
            )
            
            # Update indexed paths cache
            if collection_name not in self.indexed_paths:
                self.indexed_paths[collection_name] = set()
            self.indexed_paths[collection_name].add(relative_path)
            
            print(f"Stored embedding in Qdrant for {relative_path}")
            
        except Exception as e:
            print(f"Error indexing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.current_file = None
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
    
    async def get_all_images(self, folder_path: Optional[str] = None) -> List[Dict]:
        """Get all indexed images, optionally filtered by folder"""
        try:
            results = []
            
            if folder_path:
                # Get images from specific folder
                collection_name = self.folder_manager.get_collection_for_path(folder_path)
                if collection_name:
                    results.extend(await self._get_images_from_collection(collection_name, folder_path))
            else:
                # Get images from all folders
                for folder_info in self.folder_manager.get_all_folders():
                    if folder_info["is_valid"]:  # Only include images from valid folders
                        results.extend(await self._get_images_from_collection(
                            folder_info["collection_name"],
                            folder_info["path"]
                        ))
            
            # Sort by indexed_at timestamp (newest first)
            results.sort(key=lambda x: x["indexed_at"], reverse=True)
            return results
            
        except Exception as e:
            print(f"Error getting images: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _get_images_from_collection(self, collection_name: str, root_folder: str) -> List[Dict]:
        """Get images from a specific collection"""
        try:
            response = self.qdrant.scroll(
                collection_name=collection_name,
                limit=10000,
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
                        "root_folder": point.payload["root_folder"],  # Root folder path
                        "indexed_at": point.payload.get("indexed_at", 0)  # Include timestamp
                    }
            
            return list(unique_images.values())
            
        except Exception as e:
            print(f"Error getting images from collection {collection_name}: {e}")
            return []

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: ImageIndexer, root_folder: Path):
        self.indexer = indexer
        self.root_folder = root_folder
    
    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(self.indexer.index_image(Path(event.src_path), self.root_folder)) 