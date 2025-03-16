import torch
import numpy as np
import faiss
from PIL import Image
import sqlite3
from pathlib import Path
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel

class ImageSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize FAISS index for cosine similarity
        self.dimension = 512  # CLIP's output dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors = cosine similarity
        
        # Load existing embeddings into FAISS
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load existing embeddings from SQLite into FAISS"""
        print("Loading embeddings into FAISS index...")
        with sqlite3.connect("images.db") as conn:
            cursor = conn.execute("SELECT embedding FROM images")
            embeddings = []
            for row in cursor:
                embedding = np.frombuffer(row[0], dtype=np.float32)
                embeddings.append(embedding)
            
            if embeddings:
                embeddings = np.vstack(embeddings)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings)
                print(f"Loaded {len(embeddings)} embeddings into FAISS index")
            else:
                print("No embeddings found in database")
    
    async def search_by_text(self, query: str, k: int = 10) -> List[Dict]:
        """Search images by text query"""
        # Generate text embedding
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize the features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features.cpu().numpy()
        
        # Search similar images
        D, I = self.index.search(text_embedding, k)
        
        # Get image paths for results
        with sqlite3.connect("images.db") as conn:
            cursor = conn.execute("SELECT path FROM images")
            paths = [row[0] for row in cursor.fetchall()]
            
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx < len(paths):
                    # Convert inner product score to similarity (ranges from -1 to 1)
                    similarity = float(score)
                    # Convert to percentage (0 to 100)
                    similarity_pct = (similarity + 1) * 50
                    results.append({
                        "path": paths[idx],
                        "similarity": similarity_pct
                    })
            
            # Sort by similarity in descending order
            results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    async def search_by_image(self, image: Image.Image, k: int = 10) -> List[Dict]:
        """Search images by similarity to uploaded image"""
        # Generate image embedding
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_embedding = image_features.cpu().numpy()
        
        # Search similar images
        D, I = self.index.search(image_embedding, k)
        
        # Get image paths for results
        with sqlite3.connect("images.db") as conn:
            cursor = conn.execute("SELECT path FROM images")
            paths = [row[0] for row in cursor.fetchall()]
            
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx < len(paths):
                    # Convert inner product score to similarity (ranges from -1 to 1)
                    similarity = float(score)
                    # Convert to percentage (0 to 100)
                    similarity_pct = (similarity + 1) * 50
                    results.append({
                        "path": paths[idx],
                        "similarity": similarity_pct
                    })
            
            # Sort by similarity in descending order
            results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results 