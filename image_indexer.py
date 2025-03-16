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
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ImageIndexer:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize CLIP model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize database
        self.init_database()
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def init_database(self):
        """Initialize SQLite database for storing image metadata"""
        with sqlite3.connect("images.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def start_monitoring(self):
        """Start monitoring the data directory for changes"""
        event_handler = ImageEventHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.data_dir), recursive=False)
        observer.start()
    
    async def index_existing_images(self):
        """Index all existing images in the data directory"""
        print("Starting to index existing images...")
        image_files = [f for f in self.data_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}]
        total = len(image_files)
        for i, image_file in enumerate(image_files, 1):
            print(f"Indexing image {i}/{total}: {image_file.name}")
            await self.index_image(image_file)
        print("Finished indexing all images.")
    
    async def index_image(self, image_path: Path):
        """Index a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten()
            
            # Store in database
            with sqlite3.connect("images.db") as conn:
                embedding_blob = embedding.tobytes()
                conn.execute(
                    "INSERT OR REPLACE INTO images (path, embedding) VALUES (?, ?)",
                    (str(image_path), embedding_blob)
                )
                conn.commit()
        except Exception as e:
            print(f"Error indexing image {image_path}: {e}")
    
    async def get_all_images(self) -> List[Dict]:
        """Get all indexed images"""
        with sqlite3.connect("images.db") as conn:
            cursor = conn.execute("SELECT path, created_at FROM images")
            return [
                {"path": row[0], "created_at": row[1]}
                for row in cursor.fetchall()
            ]

class ImageEventHandler(FileSystemEventHandler):
    def __init__(self, indexer: ImageIndexer):
        self.indexer = indexer
    
    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(self.indexer.index_image(Path(event.src_path))) 