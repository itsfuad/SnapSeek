import os
import time
from pathlib import Path
from typing import List, Dict
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

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
        self.qdrant = QdrantClient(":memory:")  # Use in-memory storage for simplicity
        self.init_collection()
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def init_collection(self):
        """Initialize Qdrant collection for storing image vectors"""
        try:
            # Create collection if it doesn't exist
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"Initialized Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
    
    def start_monitoring(self):
        """Start monitoring the data directory for changes"""
        event_handler = ImageEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.data_dir), recursive=False)
        observer.start()
        print("Started monitoring data directory for changes")
    
    async def index_existing_images(self):
        """Index all existing images in the data directory"""
        print("Starting to index existing images...")
        image_files = [f for f in self.data_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
        total = len(image_files)
        print(f"Found {total} images to index")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Indexing image {i}/{total}: {image_file.name}")
            await self.index_image(image_file)
        
        print("Finished indexing all images")
    
    async def index_image(self, image_path: Path):
        """Index a single image"""
        try:
            print(f"Indexing image: {image_path}")
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
                        payload={"path": str(image_path)}
                    )
                ]
            )
            print(f"Stored embedding in Qdrant for {image_path}")
            
        except Exception as e:
            print(f"Error indexing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
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
                {"path": point.payload["path"]}
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