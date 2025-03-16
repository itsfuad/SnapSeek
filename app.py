import os
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import io

from image_indexer import ImageIndexer
from image_search import ImageSearch

app = FastAPI(title="Image Search Engine")

# Create necessary directories
UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Initialize image indexer and searcher
indexer = ImageIndexer()
searcher = ImageSearch()

@app.on_event("startup")
async def startup_event():
    """Initialize the image indexer and start monitoring the data directory"""
    indexer.start_monitoring()
    await indexer.index_existing_images()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 