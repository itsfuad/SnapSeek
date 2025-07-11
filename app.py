import os
from pathlib import Path
from typing import List, Optional
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from image_indexer import ImageIndexer
from image_search import ImageSearch

# Initialize image indexer and searcher
indexer = ImageIndexer()
searcher = ImageSearch()

image_extensions = [".jpg", ".jpeg", ".png", ".gif"]

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the image indexer"""
    yield

app = FastAPI(title="Image Search Engine", lifespan=lifespan)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    folders = indexer.folder_manager.get_all_folders()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "initial_status": {
                "status": indexer.status.value,
                "current_file": indexer.current_file,
                "total_files": indexer.total_files,
                "processed_files": indexer.processed_files,
                "progress_percentage": round((indexer.processed_files / indexer.total_files * 100) if indexer.total_files > 0 else 0, 2)
            },
            "folders": folders
        }
    )

@app.post("/folders")
async def add_folder(folder_path: str, background_tasks: BackgroundTasks):
    """Add a new folder to index"""
    try:
        # Add folder to manager first (this creates the collection)
        folder_info = indexer.folder_manager.add_folder(folder_path)
        
        # Start indexing in the background
        background_tasks.add_task(indexer.index_folder, folder_path)
        
        return folder_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.delete("/folders/{folder_path:path}")
async def remove_folder(folder_path: str):
    """Remove a folder from indexing"""
    try:
        await indexer.remove_folder(folder_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.get("/folders")
async def list_folders():
    """List all indexed folders"""
    return indexer.folder_manager.get_all_folders()

@app.get("/search/text")
async def search_by_text(query: str, folder: Optional[str] = None) -> List[dict]:
    """Search images by text query, optionally filtered by folder"""
    results = await searcher.search_by_text(query, folder)
    return results

@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    folder: Optional[str] = None
) -> List[dict]:
    """Search images by uploading a similar image, optionally filtered by folder"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = await searcher.search_by_image(image, folder)
    return results

@app.get("/images")
async def list_images(folder: Optional[str] = None) -> List[dict]:
    """List all indexed images, optionally filtered by folder"""
    return await indexer.get_all_images(folder)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time indexing status updates"""
    await indexer.add_websocket_connection(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await indexer.remove_websocket_connection(websocket)

@app.get("/thumbnail/{folder_path:path}/{file_path:path}")
async def serve_thumbnail(folder_path: str, file_path: str):
    """Serve resized image thumbnails"""
    try:
        # Get folder info to verify it's an indexed folder
        folder_info = indexer.folder_manager.get_folder_info(folder_path)
        if not folder_info:
            raise HTTPException(status_code=404, detail="Folder not found")

        # Construct full file path
        full_path = Path(folder_path) / file_path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Only serve image files
        if full_path.suffix.lower() not in image_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Open image, resize, and convert to JPEG
        img = Image.open(full_path)
        img.thumbnail((200, 200))  # Resize maintaining aspect ratio

        # Save to a byte stream
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})  # Cache for 1 hour
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{folder_path:path}/{file_path:path}")
async def serve_file(folder_path: str, file_path: str):
    """Serve files from indexed folders"""
    try:
        # Get folder info to verify it's an indexed folder
        folder_info = indexer.folder_manager.get_folder_info(folder_path)
        if not folder_info:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Construct full file path
        full_path = Path(folder_path) / file_path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Only serve image files
        if full_path.suffix.lower() not in image_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        return FileResponse(full_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

def get_windows_drives():
    """Get available drives on Windows"""
    from ctypes import windll
    drives = []
    bitmask = windll.kernel32.GetLogicalDrives()
    for letter in range(65, 91):  # A-Z
        if bitmask & (1 << (letter - 65)):
            drives.append(chr(letter) + ":\\")
    return drives

def get_directory_item(item):
    """Get directory item info"""
    try:
        is_dir = item.is_dir()
        if is_dir or item.suffix.lower() in image_extensions:
            return {
                "name": item.name,
                "path": str(item.absolute()),
                "type": "directory" if is_dir else "file",
                "size": item.stat().st_size if not is_dir else None
            }
    except Exception:
        pass
    return None

def get_directory_contents(path: str):
    """Get contents of a directory"""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return {"error": "Path does not exist"}
        
        parent = str(path_obj.parent) if path_obj.parent != path_obj else None
        contents = [
            item for item in (get_directory_item(i) for i in path_obj.iterdir())
            if item is not None
        ]
        
        return {
            "current_path": str(path_obj.absolute()),
            "parent_path": parent,
            "contents": sorted(contents, key=lambda x: (x["type"] != "directory", x["name"].lower()))
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/browse")
async def browse_folders():
    """Browse system folders"""
    if os.name == "nt":  # Windows
        return {"drives": get_windows_drives()}
    return get_directory_contents("/")  # Unix-like

@app.get("/browse/{path:path}")
async def browse_path(path: str):
    """Browse a specific path"""
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        
        # Get parent directory for navigation
        parent = str(path_obj.parent) if path_obj.parent != path_obj else None
        
        # List directories and files
        contents = []
        for item in path_obj.iterdir():
            try:
                is_dir = item.is_dir()
                if is_dir or item.suffix.lower() in image_extensions:
                    contents.append({
                        "name": item.name,
                        "path": str(item.absolute()),
                        "type": "directory" if is_dir else "file",
                        "size": item.stat().st_size if not is_dir else None
                    })
            except Exception:
                continue  # Skip items we can't access
        
        return {
            "current_path": str(path_obj.absolute()),
            "parent_path": parent,
            "contents": sorted(contents, key=lambda x: (x["type"] != "directory", x["name"].lower()))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False) 