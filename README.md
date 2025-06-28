# Arabic Legal RAG System with Agentic Workflow

## Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system with an agentic workflow designed specifically for Arabic legal documents and business registration compliance. The system uses multiple AI agents to collect information, search legal databases, and provide comprehensive recommendations.

## Key Features

### 🤖 Agentic Workflow
- **Information Collection Agent**: Gathers detailed business information from clients
- **Search Planning Agent**: Identifies key legal search points based on collected information  
- **RAG Search Agent**: Performs targeted searches in the legal document database
- **Recommendation Agent**: Provides comprehensive legal recommendations and next steps

### 📄 Advanced Document Processing
- **Tesseract OCR**: Primary text extraction from PDF documents
- **Auto-correction**: Spell checking and text correction using advanced models
- **Multi-language Translation**: Automatic detection and translation to Arabic
- **Smart Chunking**: Intelligent text segmentation for optimal retrieval

### 🔍 Intelligent Information Gathering
The system collects critical information for business registration:

1. **Company Structure**: Legal entity type, shareholders, partners
2. **Financial Information**: Expected revenue, capital, tax obligations
3. **Business Activities**: Main activities, services vs goods, international operations
4. **Tax History**: Previous registrations, pending liabilities
5. **Compliance Documentation**: Required documents, deadlines, requirements

### 🧠 Groq-Powered AI
- Uses **meta-llama/llama-3.1-70b-versatile** model via Groq API
- Fast, reliable AI responses with specialized legal prompting
- Multi-agent coordination for comprehensive analysis

### 💻 Simplified Interface
- Clean chat-based interface for natural conversation
- Real-time document upload and processing
- Structured display of agent workflow results
- Mobile-responsive design

## Technical Architecture

```
User Input → Information Collection Agent
    ↓
Search Planning Agent → Identifies Key Legal Points
    ↓
RAG Search Agent → Searches Legal Database
    ↓
Recommendation Agent → Final Legal Advice
```

## Installation & Setup

### Prerequisites
```bash
# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Set Groq API key
export GROQ_API_KEY='your-groq-api-key-here'

# Start server
python run.py
```

### Frontend Setup
```bash
npm install
npm run dev
```

### Get Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Create a free account
3. Generate an API key
4. Set the environment variable

## API Endpoints

### Chat with Agent
```http
POST /api/chat
{
    "message": "أريد تسجيل شركة ذات مسؤولية محدودة"
}
```

### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data
file: [PDF file]
```

### Health Check
```http
GET /api/health
```

## Agentic Workflow Details

### 1. Information Collection Agent
Collects essential business information:
- Company structure and ownership
- Financial details and capital
- Business activities and scope
- Tax history and compliance status
- Required documentation

### 2. Search Planning Agent
Analyzes collected information to identify:
- Relevant legal areas to research
- Priority levels for different searches
- Specific legal categories (tax, commercial, labor)

### 3. RAG Search Agent
Performs targeted searches:
- Semantic search in legal document database
- Retrieves relevant law articles and regulations
- Summarizes findings for each search area

### 4. Recommendation Agent
Provides comprehensive advice:
- Step-by-step registration process
- Legal and tax requirements
- Required documentation checklist
- Important deadlines and warnings

## Document Processing Pipeline

1. **PDF Upload**: User uploads legal documents
2. **Tesseract OCR**: Extract text from PDF pages
3. **Auto-correction**: Fix spelling and grammar errors
4. **Language Detection**: Identify source language
5. **Translation**: Convert to Arabic if needed
6. **Text Cleaning**: Normalize and clean Arabic text
7. **Chunking**: Split into searchable segments
8. **Embedding**: Create vector representations
9. **Indexing**: Add to FAISS search index

## Configuration

### Environment Variables
```bash
GROQ_API_KEY=your-groq-api-key-here
```

### Model Configuration
- **Groq Model**: meta-llama/llama-3.1-70b-versatile
- **Embedding Model**: paraphrase-multilingual-MiniLM-L12-v2
- **Chunk Size**: 1000 characters
- **Overlap**: 100 characters

## File Structure

```
├── backend/
│   ├── main.py              # FastAPI application with agentic workflow
│   ├── requirements.txt     # Python dependencies
│   └── run.py              # Startup script
├── src/
│   ├── components/
│   │   └── ChatInterface.tsx # Main chat interface
│   ├── types/
│   │   └── index.ts         # TypeScript types
│   └── App.tsx             # Main application
└── README.md
```

## Usage Examples

### Business Registration Query
```
User: "أريد تسجيل شركة ذات مسؤولية محدودة في مجال التكنولوجيا"

System Response:
1. Information Collection: Gathers company details, capital, partners
2. Search Planning: Identifies relevant legal areas
3. RAG Search: Finds applicable laws and regulations
4. Recommendations: Provides step-by-step guidance
```

### Document Upload
```
User uploads: "قانون الشركات الجديد.pdf"

System:
1. Extracts text using Tesseract OCR
2. Auto-corrects and translates if needed
3. Adds to searchable database
4. Confirms successful processing
```

## Legal Domains Covered

- **Commercial Law**: Company registration, business licenses
- **Tax Law**: Corporate taxes, VAT, withholding tax
- **Labor Law**: Employment regulations, worker rights
- **Investment Law**: Foreign investment, incentives
- **Regulatory Compliance**: Industry-specific regulations

## Limitations

- Optimized for Egyptian legal system
- Requires Groq API key for full functionality
- PDF processing depends on Tesseract OCR quality
- Arabic language focus (though supports translation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the documentation
2. Review API endpoints at `/docs`
3. Ensure all dependencies are installed
4. Verify Groq API key is set correctly

## Citation

```bibtex
@software{arabic_legal_rag_agentic_2024,
    title={Arabic Legal RAG System with Agentic Workflow},
    author={[Your Name]},
    year={2024},
    url={[Repository URL]}
}
```