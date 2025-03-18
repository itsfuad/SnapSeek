from pathlib import Path
from typing import List, Dict, Optional
import json
import time
from qdrant_singleton import QdrantClientSingleton

class FolderManager:
    def __init__(self):
        # Ensure config directory exists
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Ensure folders.json exists
        self.config_file = self.config_dir / "folders.json"
        if not self.config_file.exists():
            self._create_default_config()
        
        self.folders: Dict[str, Dict] = self._load_folders()
        
    def _create_default_config(self):
        """Create default configuration file if it doesn't exist"""
        default_config = {}
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default configuration file at {self.config_file}")
    
    def _load_folders(self) -> Dict[str, Dict]:
        """Load folder configurations from JSON file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_folders(self):
        """Save folder configurations to JSON file"""
        # Ensure config directory exists before saving
        self.config_dir.mkdir(exist_ok=True)
        
        # Write config
        with open(self.config_file, 'w') as f:
            json.dump(self.folders, f, indent=2)
    
    def add_folder(self, folder_path: str) -> Dict:
        """Add a new folder to index"""
        folder_path = str(Path(folder_path).absolute())
        print(f"Adding folder: {folder_path}")
        
        # Check if this folder or any parent/child is already being indexed
        for existing_path in self.folders:
            existing = Path(existing_path)
            new_path = Path(folder_path)
            
            # If the new path is already indexed
            if existing == new_path:
                print(f"Folder already indexed: {folder_path}")
                return self.folders[existing_path]
            
            # If the new path is a parent of an existing path, use the same collection
            if existing.is_relative_to(new_path):
                print(f"Using existing collection for parent path: {folder_path}")
                return self.folders[existing_path]
            
            # If the new path is a child of an existing path, use the parent's collection
            if new_path.is_relative_to(existing):
                print(f"Using parent's collection for: {folder_path}")
                return self.folders[existing_path]
        
        # If it's a completely new path, create a new entry
        collection_name = f"images_{len(self.folders)}"
        print(f"Creating new collection {collection_name} for folder: {folder_path}")
        
        folder_info = {
            "path": folder_path,
            "collection_name": collection_name,
            "added_at": int(time.time()),
            "last_indexed": None
        }
        
        # Initialize new collection in Qdrant
        QdrantClientSingleton.initialize_collection(collection_name)
        
        # Save to config
        self.folders[folder_path] = folder_info
        self._save_folders()
        
        print(f"Successfully added folder {folder_path} with collection {collection_name}")
        return folder_info
    
    def remove_folder(self, folder_path: str):
        """Remove a folder from indexing"""
        folder_path = str(Path(folder_path).absolute())
        if folder_path in self.folders:
            # Delete the collection
            collection_name = self.folders[folder_path]["collection_name"]
            client = QdrantClientSingleton.get_instance()
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception as e:
                print(f"Error deleting collection: {e}")
            
            # Remove from config
            del self.folders[folder_path]
            self._save_folders()
    
    def get_folder_info(self, folder_path: str) -> Optional[Dict]:
        """Get information about an indexed folder"""
        folder_path = str(Path(folder_path).absolute())
        return self.folders.get(folder_path)
    
    def get_all_folders(self) -> List[Dict]:
        """Get all indexed folders"""
        return [
            {
                "path": path,
                **info,
                "is_valid": Path(path).exists()  # Check if folder still exists
            }
            for path, info in self.folders.items()
        ]
    
    def update_last_indexed(self, folder_path: str):
        """Update the last indexed timestamp for a folder"""
        folder_path = str(Path(folder_path).absolute())
        if folder_path in self.folders:
            self.folders[folder_path]["last_indexed"] = int(time.time())
            self._save_folders()
    
    def get_collection_for_path(self, folder_path: str) -> Optional[str]:
        """Get the collection name for a given path"""
        folder_path = Path(folder_path).absolute()
        print(f"Looking for collection for path: {folder_path}")
        
        # Check each indexed folder to find the appropriate collection
        for path, info in self.folders.items():
            if folder_path == Path(path) or folder_path.is_relative_to(Path(path)):
                print(f"Found collection {info['collection_name']} for path {folder_path}")
                return info["collection_name"]
        
        print(f"No collection found for path {folder_path}")
        return None 