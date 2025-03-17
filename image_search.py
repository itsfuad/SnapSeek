import torch
from PIL import Image
from typing import List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel
from qdrant_singleton import QdrantClientSingleton
from folder_manager import FolderManager

class ImageSearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize Qdrant client and folder manager
        self.qdrant = QdrantClientSingleton.get_instance()
        self.folder_manager = FolderManager()
    
    def calculate_similarity_percentage(self, score: float) -> float:
        """Convert cosine similarity score to percentage"""
        # Qdrant returns cosine similarity scores between -1 and 1
        # We want to convert this to a percentage between 0 and 100
        # First normalize to 0-1 range, then convert to percentage
        normalized = (score + 1) / 2
        return normalized * 100

    def filter_results(self, search_results: list, threshold: float = 0.5) -> List[Dict]:
        """Filter and format search results"""
        results = []
        for scored_point in search_results:
            # Convert cosine similarity to percentage
            similarity = self.calculate_similarity_percentage(scored_point.score)
            
            # Only include results above threshold (50% similarity)
            if similarity >= threshold:
                results.append({
                    "path": scored_point.payload["path"],
                    "absolute_path": scored_point.payload["absolute_path"],
                    "root_folder": scored_point.payload["root_folder"],
                    "similarity": round(similarity, 1)  # Round to 1 decimal place
                })
        
        return results
    
    async def search_by_text(self, query: str, folder_path: Optional[str] = None, k: int = 10) -> List[Dict]:
        """Search images by text query"""
        try:
            print(f"\nSearching for text: '{query}'")
            
            # Get collections to search
            collections_to_search = []
            if folder_path:
                # Search in specific folder's collection
                collection_name = self.folder_manager.get_collection_for_path(folder_path)
                if collection_name:
                    collections_to_search.append(collection_name)
            else:
                # Search in all collections
                folders = self.folder_manager.get_all_folders()
                collections_to_search.extend(folder["collection_name"] for folder in folders if folder["is_valid"])
            
            if not collections_to_search:
                print("No collections available to search")
                return []
            
            # Generate text embedding
            inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy().flatten()
            
            # Search in all relevant collections
            all_results = []
            for collection_name in collections_to_search:
                try:
                    # Get more results from each collection when searching multiple collections
                    collection_limit = k * 3 if len(collections_to_search) > 1 else k
                    
                    search_result = self.qdrant.search(
                        collection_name=collection_name,
                        query_vector=text_embedding.tolist(),
                        limit=collection_limit,  # Get more results from each collection
                        offset=0,  # Explicitly set offset
                        score_threshold=0.0  # We'll filter results ourselves
                    )
                    
                    # Filter and format results
                    results = self.filter_results(search_result, threshold=50)
                    all_results.extend(results)
                    print(f"Found {len(results)} matches in collection {collection_name}")
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort all results by similarity
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Take top k results
            final_results = all_results[:k]
            print(f"Found {len(final_results)} total relevant matches across {len(collections_to_search)} collections")
            
            return final_results
            
        except Exception as e:
            print(f"Error in text search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def search_by_image(self, image: Image.Image, folder_path: Optional[str] = None, k: int = 10) -> List[Dict]:
        """Search images by similarity to uploaded image"""
        try:
            # Get collections to search
            collections_to_search = []
            if folder_path:
                # Search in specific folder's collection
                collection_name = self.folder_manager.get_collection_for_path(folder_path)
                if collection_name:
                    collections_to_search.append(collection_name)
            else:
                # Search in all collections
                folders = self.folder_manager.get_all_folders()
                collections_to_search.extend(folder["collection_name"] for folder in folders if folder["is_valid"])
            
            if not collections_to_search:
                print("No collections available to search")
                return []
            
            # Generate image embedding
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.cpu().numpy().flatten()
            
            # Search in all relevant collections
            all_results = []
            for collection_name in collections_to_search:
                try:
                    # Get more results from each collection when searching multiple collections
                    collection_limit = k * 3 if len(collections_to_search) > 1 else k
                    
                    search_result = self.qdrant.search(
                        collection_name=collection_name,
                        query_vector=image_embedding.tolist(),
                        limit=collection_limit,  # Get more results from each collection
                        offset=0,  # Explicitly set offset
                        score_threshold=0.0  # We'll filter results ourselves
                    )
                    
                    # Filter and format results
                    results = self.filter_results(search_result, threshold=50)
                    all_results.extend(results)
                    print(f"Found {len(results)} matches in collection {collection_name}")
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort all results by similarity
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Take top k results
            final_results = all_results[:k]
            print(f"Found {len(final_results)} total relevant matches across {len(collections_to_search)} collections")
            
            return final_results
            
        except Exception as e:
            print(f"Error in image search: {e}")
            import traceback
            traceback.print_exc()
            return [] 