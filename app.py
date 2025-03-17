from fastapi import FastAPI, File, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List
from PIL import Image
import io
import asyncio
from contextlib import asynccontextmanager

from image_indexer import ImageIndexer
from image_search import ImageSearch

# Initialize image indexer and searcher
indexer = ImageIndexer()
searcher = ImageSearch()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the image indexer and start monitoring the data directory"""
    # Start monitoring immediately
    indexer.start_monitoring()
    # Start indexing in the background without waiting
    asyncio.create_task(indexer.index_existing_images())
    yield

app = FastAPI(title="Image Search Engine", lifespan=lifespan)

# Create necessary directories
UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
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
            }
        }
    )

@app.get("/search/text")
async def search_by_text(query: str) -> List[dict]:
    """Search images by text query"""
    results = await searcher.search_by_text(query)
    return results

@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...)) -> List[dict]:
    """Search images by uploading a similar image"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = await searcher.search_by_image(image)
    return results

@app.get("/images")
async def list_images() -> List[dict]:
    """List all indexed images with their metadata"""
    return await indexer.get_all_images()

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time indexing status updates"""
    await indexer.add_websocket_connection(websocket)
    try:
        while True:
            # Keep the connection alive and handle any incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        await indexer.remove_websocket_connection(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await indexer.remove_websocket_connection(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False) 