# AI-Powered Image Search Engine

A modern image search engine that uses CLIP (Contrastive Language-Image Pre-Training) for semantic search and Qdrant for vector similarity search. This system allows you to search images using either text descriptions or similar images.

## Features

- 🖼️ **Auto-Indexing**: Automatically indexes images from the `data` folder
- 🔍 **Text Search**: Find images using natural language descriptions
- 📸 **Image Search**: Find similar images by uploading a reference image
- ⚡ **Real-time Updates**: Monitors the data folder for new images and indexes them automatically
- 🎯 **High Accuracy**: Uses OpenAI's CLIP model for high-quality image-text matching
- 📊 **Similarity Scores**: Shows match percentage for each result

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd photobooth
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Your Images**:
   - Place your images in the `data` folder
   - Supported formats: JPG, JPEG, PNG, GIF

2. **Start the Server**:
```bash
python app.py
```
   - The server will start at `http://localhost:8000`
   - Images will be automatically indexed on startup

3. **Using the Interface**:
   - Open your browser and go to `http://localhost:8000`
   - Use the text search box to find images by description
   - Use the image upload to find similar images

## Project Structure

```
SnapSeek/
├── app.py              # FastAPI application
├── data/              # Image storage directory
├── image_indexer.py   # Image indexing and monitoring
├── image_search.py    # Search functionality
├── qdrant_singleton.py # Vector database client
├── requirements.txt   # Python dependencies
├── static/           # Static files
└── templates/        # HTML templates
```

## Technical Details

- **Frontend**: HTML + Tailwind CSS
- **Backend**: FastAPI
- **Image Processing**: CLIP (Contrastive Language-Image Pre-Training)
- **Vector Database**: Qdrant
- **Image Monitoring**: Watchdog

## How It Works

1. **Indexing**:
   - Images are processed through the CLIP model
   - Generates 512-dimensional feature vectors
   - Vectors are stored in Qdrant for fast similarity search

2. **Text Search**:
   - Text query is converted to a feature vector using CLIP
   - Vector similarity search finds matching images
   - Results are ranked by similarity score

3. **Image Search**:
   - Uploaded image is processed through CLIP
   - Similar images are found using vector similarity
   - Results show percentage match scores

## Performance Notes

- First-time startup might be slower due to CLIP model download
- Search performance depends on dataset size
- GPU acceleration is automatic if available

## Limitations

- Best results with a diverse image dataset
- Similarity scores may vary based on dataset size
- Memory usage increases with the number of indexed images

## Future Improvements

- Add image metadata storage
- Implement image tagging
- Add batch processing for large datasets
- Add image preprocessing options
- Implement result filtering and categorization

## Troubleshooting

1. **No Images Found**:
   - Check if images are in the `data` folder
   - Verify supported image formats
   - Check console for indexing errors

2. **Poor Search Results**:
   - Try more specific search queries
   - Add more diverse images to the dataset
   - Use image search for more precise matching

3. **Performance Issues**:
   - Consider using GPU acceleration
   - Reduce the number of indexed images
   - Adjust the similarity threshold

## License

This project is licensed under the Mozilla Public License 2.0 - see the LICENSE file for details.

## Credits

- CLIP model by OpenAI
- Qdrant vector database
- FastAPI framework 