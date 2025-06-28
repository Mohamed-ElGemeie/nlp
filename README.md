# Arabic Legal Documents RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed for Arabic legal documents. The system processes PDF legal documents, creates searchable embeddings, and provides AI-powered responses to legal queries in Arabic.

## Implemented Features

### 📄 Document Processing
- **PDF Text Extraction**: Uses PyMuPDF for direct text extraction from legal PDFs
- **OCR Support**: EasyOCR and Tesseract for scanned document processing
- **Arabic Text Cleaning**: Specialized preprocessing for Arabic legal text
- **Smart Chunking**: Segments documents based on legal structure (المادة, الفقرة, البند)
- **Progress Tracking**: Shows processing progress (e.g., "Processing document 5/20")

### 🔍 Search and Retrieval
- **Semantic Search**: Uses sentence-transformers multilingual model for Arabic understanding
- **Vector Database**: FAISS IndexFlatIP for similarity search
- **Embedding Caching**: Saves embeddings to `embeddinghere.pkl` to avoid reprocessing
- **Context Retrieval**: Returns relevant document chunks with source metadata

### 🤖 AI Integration
- **Ollama Support**: Local LLM integration (qwen2.5:14b, llama3:latest)
- **Arabic Responses**: Configured to respond only in Arabic
- **Legal Context**: Specialized prompting for Egyptian legal framework

### 💻 Web Interface
- **FastAPI Backend**: REST API with automatic documentation
- **React Frontend**: TypeScript-based user interface with search functionality
- **Health Monitoring**: System status and statistics endpoints

## Technical Architecture

```
Frontend (React/TypeScript)
    ↓ HTTP Requests
Backend (FastAPI)
    ↓ PDF Processing
PyMuPDF + OCR Engines
    ↓ Text Embedding
Sentence Transformers
    ↓ Vector Storage
FAISS Index
    ↓ AI Generation
Ollama (Local LLM)
```

## Core Components

### Document Processing Pipeline
```python
# Text extraction with fallback to OCR
def extract_text_from_legal_pdf(pdf_path: str) -> str:
    # 1. Try direct text extraction
    # 2. Fall back to structured extraction 
    # 3. Use OCR for scanned pages
    # 4. Clean and validate Arabic text
```

### Arabic Text Processing
```python
# Specialized Arabic legal text cleaning
def advanced_arabic_text_cleaner(text: str) -> str:
    # 1. Unicode normalization
    # 2. Arabic character standardization
    # 3. Legal document formatting fixes
    # 4. Quality filtering
```

### Embedding System
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Storage**: FAISS IndexFlatIP with cosine similarity
- **Caching**: Persistent embeddings storage with pickle

## Installation

### Requirements
```bash
# Backend dependencies
pip install -r backend/requirements.txt

# Frontend dependencies  
npm install
```

### Backend Setup
```bash
cd backend
python main.py
# Runs on http://localhost:8000
```

### Frontend Setup
```bash
npm run dev
# Runs on http://localhost:5173
```

### AI Models
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models
ollama pull qwen2.5:14b
ollama pull llama3:latest
```

## API Endpoints

### Search Documents
```http
POST /api/search/stream
{
    "query": "ما هي الضرائب المفروضة على الشركات؟",
    "top_k": 5
}
```

### System Status
```http
GET /api/health
GET /api/stats
```

## File Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── embeddinghere.pkl    # Cached embeddings (auto-generated)
├── src/
│   ├── components/          # React components
│   │   ├── SearchInterface.tsx
│   │   ├── SearchResults.tsx
│   │   ├── SystemStats.tsx
│   │   └── Header.tsx
│   └── قوانين/              # PDF legal documents
└── package.json             # Node.js dependencies
```

## How It Works

1. **Startup**: System checks for cached embeddings in `embeddinghere.pkl`
2. **First Run**: Processes all PDFs in `src/قوانين/` directory, creates embeddings, saves cache
3. **Subsequent Runs**: Loads from cache (much faster startup)
4. **Search**: User query → embedding → FAISS search → context assembly → LLM response
5. **Response**: AI generates Arabic legal answer with document citations

## Current Limitations

- **Language**: Optimized for Arabic legal documents only
- **Scope**: Egyptian legal system focus
- **Processing**: Sequential PDF processing (not parallel)
- **Search**: Basic semantic search without advanced filtering
- **UI**: Simple interface without advanced features

## Technical Specifications

### Dependencies
- **Backend**: FastAPI, PyMuPDF, sentence-transformers, faiss-cpu, easyocr
- **Frontend**: React, TypeScript, Vite
- **AI**: Ollama with qwen2.5:14b and llama3:latest models
- **Storage**: Pickle files for embedding cache

### System Requirements
- **Python**: 3.8+
- **Node.js**: 16+
- **Memory**: ~2GB for processing moderate document collections
- **Storage**: Variable based on PDF collection size

## Research Applications

This system demonstrates:
- **Arabic NLP**: Domain-specific text processing for legal documents
- **RAG Architecture**: Practical implementation of retrieval-augmented generation
- **Legal AI**: Application of AI to legal information retrieval
- **Document Processing**: Multi-modal PDF processing with OCR fallback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Test with sample legal documents
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{arabic_legal_rag_2024,
    title={Arabic Legal Documents RAG System},
    author={[Your Name]},
    year={2024},
    url={[Repository URL]}
}
``` 