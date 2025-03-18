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
git clone https://github.com/itsfuad/SnapSeek
cd SnapSeek
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the backend server:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Add your photo folders and start searching!

## Development

### Running Tests

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run the tests:
```bash
pytest tests/ -v
```

### GitHub Actions

The project includes GitHub Actions workflows for automated testing:
- Tests run on every push to main branch
- Tests run on every pull request to main branch
- Uses Python 3.11 on Ubuntu latest
- Installs dependencies from both `requirements.txt` and `requirements-test.txt`

## Project Structure

```
Project
├── main.py              # FastAPI application entry point
├── requirements.txt     # Main dependencies
├── requirements-test.txt # Test dependencies
├── templates/          # Frontend templates
├── static/            # Static assets
├── tests/             # Test files
└── .github/
    └── workflows/     # GitHub Actions workflows
```

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