# Installation Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/job-recommendation-system.git
cd job-recommendation-system
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n jobrec python=3.9
conda activate jobrec

# OR using venv
python -m venv jobrec
source jobrec/bin/activate  # Linux/Mac
# jobrec\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Download Required Data
```bash
# Create data directories
mkdir -p data/raw data/processed data/cleaned

# Download datasets (links in README.md)
# Place files in data/raw/
```

### 5. Optional: LLM Evaluation Setup
```bash
# Create .env file for OpenAI API (if using LLM evaluation)
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Detailed Setup

### System Requirements
- **Python**: 3.8+ (3.9 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for datasets and models
- **OS**: Windows, macOS, or Linux

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **sentence-transformers**: SBERT embeddings
- **nltk/spacy**: Text preprocessing
- **matplotlib/seaborn**: Visualization

### Installing spaCy Language Model
```bash
# Download English language model
python -m spacy download en_core_web_sm
```

### Jupyter Lab Setup
```bash
# Install kernel for the environment
python -m ipykernel install --user --name jobrec --display-name "Job Recommendation System"

# Launch Jupyter Lab
jupyter lab
```

## Verification

### Test Installation
```python
# Test core imports
from src.jobrec.recommender import ModelFactory
from src.jobrec.evaluator import RecommendationEvaluator
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("✅ Installation successful!")
```

### Run Quick Demo
```bash
# Navigate to notebooks and run the first few cells
cd notebooks
jupyter lab 03_modeling.ipynb
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**NLTK Data Missing:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Memory Issues:**
- Reduce batch size in configuration
- Use smaller dataset samples for testing
- Ensure sufficient RAM (16GB recommended)

**GPU Support (Optional):**
```bash
# For CUDA-enabled PyTorch (faster SBERT)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Variables
```bash
# Optional: Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0  # If using GPU
```

## Development Setup

### Additional Tools
```bash
# Code formatting and linting
pip install black flake8 pre-commit

# Testing
pip install pytest pytest-cov

# Documentation
pip install sphinx sphinx-rtd-theme
```

### Pre-commit Hooks
```bash
# Setup code quality checks
pre-commit install
```

## Docker Setup (Alternative)

### Build Container
```bash
# Build Docker image
docker build -t jobrec .

# Run container
docker run -it -p 8888:8888 jobrec
```

### Docker Compose
```bash
# Use provided docker-compose.yml
docker-compose up
```

---

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Verify system requirements
3. Ensure all dependencies are installed
4. Check Python version compatibility

For dataset-specific issues, refer to the original Kaggle sources linked in the README.md.