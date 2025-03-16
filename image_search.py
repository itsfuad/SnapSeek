import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel
from qdrant_client.http.models import SearchRequest
from qdrant_singleton import QdrantClientSingleton

class ImageSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Qdrant client
        self.collection_name = "images"
        self.qdrant = QdrantClientSingleton.get_instance()
        
        # Verify collection exists
        collections = self.qdrant.get_collections()
        if not any(col.name == self.collection_name for col in collections.collections):
            print(f"Warning: Collection '{self.collection_name}' not found. Please add some images first.")
    
    async def search_by_text(self, query: str, k: int = 10) -> List[Dict]:
        """Search images by text query"""
        try:
            print(f"\nSearching for text: '{query}'")
            
            # Generate text embedding
            inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy().flatten()
            
            # Search in Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=text_embedding.tolist(),
                limit=k
            )
            
            print(f"Found {len(search_result)} matches")
            
            # Convert results
            results = []
            for scored_point in search_result:
                similarity = (scored_point.score + 1) * 50  # Convert to percentage
                results.append({
                    "path": scored_point.payload["path"],
                    "similarity": similarity
                })
            
            return results
            
        except Exception as e:
            print(f"Error in text search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def search_by_image(self, image: Image.Image, k: int = 10) -> List[Dict]:
        """Search images by similarity to uploaded image"""
        try:
            # Generate image embedding
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.cpu().numpy().flatten()
            
            # Search in Qdrant
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=image_embedding.tolist(),
                limit=k
            )
            
            # Convert results
            results = []
            for scored_point in search_result:
                similarity = (scored_point.score + 1) * 50  # Convert to percentage
                results.append({
                    "path": scored_point.payload["path"],
                    "similarity": similarity
                })
            
            return results
            
        except Exception as e:
            print(f"Error in image search: {e}")
            import traceback
            traceback.print_exc()
            return [] 