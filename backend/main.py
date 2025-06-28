from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import glob
import fitz  # PyMuPDF
import numpy as np
import faiss
import re
import subprocess
import json
from typing import List, Dict, Optional
import asyncio
import warnings
import unicodedata
import io
import cv2
from PIL import Image
import pytesseract
import easyocr
import arabic_reshaper
from bidi.algorithm import get_display
import pickle
import time

# Sentence embeddings for better search  
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

app = FastAPI(title="Enhanced Arabic Legal Documents RAG API - Multi-Agent")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "..", "src", "قوانين")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddinghere.pkl")
CHUNK_SIZE = 1000
OVERLAP = 100
TOP_K = 5

# Multi-Agent Configuration
AGENT_MODELS = [
    "llama3:latest",
    "qwen2.5:14b"
]
ORCHESTRATOR_MODEL = "deepseek-r1:32b"

# Global variables
embedding_model = None
faiss_index = None
documents = []
easyocr_reader = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    content: str
    source: str
    metadata: str
    similarity_score: float
    chunk_id: int

class AgentResponse(BaseModel):
    model_name: str
    response: str
    processing_time: float
    error: Optional[str] = None

class MultiAgentRAGResponse(BaseModel):
    question: str
    retrieved_documents: List[SearchResult]
    agent_responses: List[AgentResponse]
    orchestrator_response: str
    final_answer: str
    total_processing_time: Optional[float] = None
    error: Optional[str] = None

class SystemStats(BaseModel):
    totalDocuments: int
    totalChunks: int
    indexSize: int
    lastUpdated: str
    processingStatus: str
    agentModels: List[str]
    orchestratorModel: str

def initialize_ocr():
    """Initialize OCR readers for Arabic text"""
    global easyocr_reader
    try:
        print("🔧 تحميل محرك OCR للنصوص العربية...")
        # Initialize EasyOCR with Arabic support
        easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
        print("✅ تم تحميل محرك OCR بنجاح")
        return True
    except Exception as e:
        print(f"❌ فشل تحميل محرك OCR: {e}")
        return False

def fix_legal_document_formatting(text: str) -> str:
    """Fix specific formatting issues in Egyptian legal documents"""
    if not text:
        return ""
    
    try:
        # Fix common legal document patterns
        # Fix article numbers and legal references
        text = re.sub(r'(\d+)\s*مكرر\s*\)\s*([أبجد])\s*\(', r'\1 مكرر (\2)', text)
        text = re.sub(r'القانون\s*رقم\s*(\d+)\s*لسنة\s*(\d+)', r'القانون رقم \1 لسنة \2', text)
        text = re.sub(r'المادة\s*(\d+)', r'المادة \1', text)
        text = re.sub(r'الفقرة\s*(\d+)', r'الفقرة \1', text)
        
        # Fix date formatting
        text = re.sub(r'(\d+)\s*([يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر])\s*سنة\s*(\d+)', r'\1 \2 سنة \3', text)
        
        # Fix Hijri months
        text = re.sub(r'([محرم|صفر|ربيع الأول|ربيع الثاني|جمادى الأولى|جمادى الآخرة|رجب|شعبان|رمضان|شوال|ذو القعدة|ذو الحجة])\s*سنة\s*(\d+)', r'\1 سنة \2', text)
        
        # Fix official gazette references
        text = re.sub(r'الجريدة\s*الرسمية\s*العدد\s*(\d+)', r'الجريدة الرسمية العدد \1', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([،؛؟!])\s*', r'\1 ', text)
        text = re.sub(r'\s*([:])\s*', r'\1 ', text)
        
        # Fix parentheses spacing
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text.strip()
        
    except Exception as e:
        print(f"خطأ في إصلاح تنسيق الوثيقة القانونية: {e}")
        return text

def advanced_arabic_text_cleaner(text: str) -> str:
    """Advanced Arabic text cleaning specifically for legal documents"""
    if not text or not text.strip():
        return ""
    
    try:
        # Normalize Unicode for proper Arabic handling
        text = unicodedata.normalize('NFKC', text)
        
        # Remove BOM and invisible characters
        text = re.sub(r'[\ufeff\u200b\u200c\u200d\u2060\u061c\u202a-\u202e]', '', text)
        
        # Fix encoding artifacts and corrupted characters
        text = re.sub(r'[]+', '', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove obviously corrupted sequences but preserve legal symbols
        text = re.sub(r'[#$%&\*\+=<>\[\]\\`\{\|~]+', ' ', text)
        
        # Keep Arabic text, numbers, punctuation, and legal symbols
        allowed_pattern = (
            r'[\u0600-\u06FF'      # Arabic
            r'\u0750-\u077F'       # Arabic Supplement  
            r'\u08A0-\u08FF'       # Arabic Extended-A
            r'\uFB50-\uFDFF'       # Arabic Presentation Forms-A
            r'\uFE70-\uFEFF'       # Arabic Presentation Forms-B
            r'\u0660-\u0669'       # Arabic-Indic digits
            r'\u06F0-\u06F9'       # Extended Arabic-Indic digits
            r'\u0020-\u007F'       # Basic Latin
            r'\u00A0-\u00FF'       # Latin-1 Supplement
            r'\n\t\r '            # Whitespace
            r'\u060C\u061B\u061F\u0640\u066A-\u066D'  # Arabic punctuation
            r'(),.:;!?\-/°%]+'     # Common punctuation and symbols
        )
        cleaned_chars = ''.join(re.findall(allowed_pattern, text))
        text = cleaned_chars
        
        # Normalize Arabic letter variants
        text = re.sub(r'[إأآا]+', 'ا', text)  # Alef variants
        text = re.sub(r'[يىئ]+', 'ي', text)   # Yeh variants  
        text = re.sub(r'[ةه]+(?=\s|$)', 'ة', text)  # Teh marbuta
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove headers/footers/page numbers
        text = re.sub(r'صفحة\s*رقم\s*\d+', '', text)
        text = re.sub(r'Page\s*\d+', '', text)
        text = re.sub(r'^[-=_\s\.]+$', '', text, flags=re.MULTILINE)
        
        # Fix Arabic punctuation
        text = re.sub(r'\s*([،؛؟!\u061F\u060C\u061B])\s*', r'\1 ', text)
        
        # Apply legal document formatting fixes
        text = fix_legal_document_formatting(text)
        
        # Quality filtering - remove very short fragments
        words = text.split()
        meaningful_words = []
        for word in words:
            # Keep words with 2+ chars or important single chars
            if len(word) >= 2 or word in ['و', 'أو', 'في', 'من', 'إلى', 'على', 'لا', '،', '؛', '؟', '!', '.']:
                meaningful_words.append(word)
        
        result = ' '.join(meaningful_words).strip()
        
        # Apply Arabic reshaping for proper display
        if result:
            try:
                reshaped = arabic_reshaper.reshape(result)
                result = str(get_display(reshaped))
            except:
                pass  # If reshaping fails, keep original
        
        return result
        
    except Exception as e:
        print(f"خطأ في تنظيف النص العربي: {e}")
        return text.strip() if text else ""

def enhanced_ocr_for_arabic(image_data: bytes) -> str:
    """Enhanced OCR specifically tuned for Arabic legal documents"""
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        img_array = np.array(image)
        
        # Preprocess for optimal Arabic OCR
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Multiple preprocessing approaches for legal documents
        processed_images = []
        
        # Original enhanced image
        height, width = gray.shape
        if height < 1200:  # Higher resolution for legal text
            scale = 1200 / height
            new_width = int(width * scale)
            enhanced = cv2.resize(gray, (new_width, 1200), interpolation=cv2.INTER_LANCZOS4)
        else:
            enhanced = gray
            
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Adaptive threshold for text clarity
        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        processed_images.append(('adaptive', adaptive))
        
        # Morphological operations for text enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        processed_images.append(('morphological', morph))
        
        best_results = []
        
        for method_name, processed_img in processed_images:
            # EasyOCR - usually better for Arabic
            if easyocr_reader:
                try:
                    easy_results = easyocr_reader.readtext(processed_img, detail=0, paragraph=True, width_ths=0.7, height_ths=0.7)
                    if easy_results:
                        text_parts = []
                        for result in easy_results:
                            if isinstance(result, str):
                                text_parts.append(result)
                            elif isinstance(result, (list, tuple)) and len(result) > 0:
                                text_parts.append(str(result[0] if hasattr(result, '__getitem__') else result))
                        if text_parts:
                            easy_text = ' '.join(text_parts)
                            best_results.append((f'EasyOCR-{method_name}', easy_text))
                except Exception as e:
                    print(f"خطأ في EasyOCR {method_name}: {e}")
            
            # Tesseract with Arabic optimization
            try:
                # Multiple Tesseract configurations for Arabic legal text
                configs = [
                    '--oem 3 --psm 6 -l ara',  # Standard Arabic
                    '--oem 3 --psm 4 -l ara',  # Single column
                    '--oem 3 --psm 3 -l ara',  # Fully automatic
                ]
                
                for i, config in enumerate(configs):
                    try:
                        tess_text = pytesseract.image_to_string(processed_img, config=config)
                        if tess_text.strip():
                            best_results.append((f'Tesseract-{method_name}-{i}', tess_text))
                            break  # Use first successful result
                    except:
                        continue
            except Exception as e:
                print(f"خطأ في Tesseract {method_name}: {e}")
        
        # Select best result based on quality metrics
        if best_results:
            best_text = ""
            best_score = 0
            
            for method, text in best_results:
                cleaned = advanced_arabic_text_cleaner(text)
                if cleaned and len(cleaned.strip()) > 20:
                    # Score based on Arabic content and length
                    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', cleaned))
                    total_chars = len(cleaned.replace(' ', ''))
                    
                    if total_chars > 0:
                        arabic_ratio = arabic_chars / total_chars
                        length_score = min(len(cleaned) / 1000, 1.0)  # Normalize length
                        quality_score = arabic_ratio * 0.7 + length_score * 0.3
                        
                        if quality_score > best_score:
                            best_score = quality_score
                            best_text = cleaned
                            print(f"✅ أفضل OCR من {method}: نسبة عربية {arabic_ratio:.2f}, طول {len(cleaned)}")
            
            return best_text
        
        return ""
        
    except Exception as e:
        print(f"خطأ في OCR المحسن: {e}")
        return ""

def initialize_embedding_model():
    """Initialize Arabic-compatible embedding model"""
    global embedding_model
    
    try:
        print("📥 تحميل نموذج التشفير للبحث...")
        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ تم تحميل نموذج التشفير بنجاح")
        return True
        
    except Exception as e:
        print(f"❌ فشل تحميل نموذج التشفير: {e}")
        return False

def save_embeddings_and_documents(embeddings_array):
    """Save embeddings and documents to file"""
    global documents
    
    try:
        print("💾 حفظ التضمينات والوثائق...")
        
        # Prepare data to save
        save_data = {
            'embeddings': embeddings_array,
            'documents': documents,
            'index_size': len(embeddings_array)
        }
        
        # Save to pickle file
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ تم حفظ {len(documents)} وثيقة و {len(embeddings_array)} تضمين في {EMBEDDINGS_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في حفظ التضمينات: {e}")
        return False

def load_embeddings_and_documents():
    """Load embeddings and documents from file"""
    global faiss_index, documents
    
    try:
        if not os.path.exists(EMBEDDINGS_FILE):
            print("📁 لا يوجد ملف تضمينات محفوظ")
            return False
        
        print("📂 تحميل التضمينات والوثائق المحفوظة...")
        
        # Load data from pickle file
        with open(EMBEDDINGS_FILE, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore documents
        documents = save_data['documents']
        
        # Restore FAISS index
        embeddings = save_data['embeddings']
        dimension = embeddings.shape[1]
        
        faiss_index = faiss.IndexFlatIP(dimension)
        if len(embeddings.shape) > 1:
            faiss_index.add(embeddings.astype('float32'))
        else:
            faiss_index.add(embeddings.reshape(1, -1).astype('float32'))
        
        print(f"✅ تم تحميل {len(documents)} وثيقة و {faiss_index.ntotal} تضمين بنجاح")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في تحميل التضمينات: {e}")
        return False

def extract_text_from_legal_pdf(pdf_path: str) -> str:
    """Enhanced PDF text extraction for Arabic legal documents"""
    try:
        print(f"📄 معالجة الوثيقة القانونية: {os.path.basename(pdf_path)}")
        
        # Use proper PyMuPDF API
        doc = fitz.Document(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = ""
            
            # Method 1: Direct text extraction with Arabic optimization
            try:
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                
                if text and len(text.strip()) > 50:
                    # Quick quality check for Arabic legal content
                    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
                    if arabic_chars > 15:  # Sufficient Arabic content
                        print(f"✅ استخراج مباشر ناجح للصفحة {page_num + 1}")
                    else:
                        text = ""  # Try other methods
                        
            except Exception as e:
                print(f"خطأ في الاستخراج المباشر للصفحة {page_num + 1}: {e}")
                text = ""
            
            # Method 2: Structured extraction for better layout preservation  
            if not text or len(text.strip()) < 50:
                try:
                    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)
                    extracted_lines = []
                    
                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            block_text = []
                            for line in block["lines"]:
                                line_text = ""
                                for span in line["spans"]:
                                    span_text = span.get("text", "").strip()
                                    if span_text:
                                        line_text += span_text + " "
                                if line_text.strip():
                                    block_text.append(line_text.strip())
                            
                            if block_text:
                                # Join lines in block, preserving structure
                                block_content = " ".join(block_text)
                                extracted_lines.append(block_content)
                    
                    if extracted_lines:
                        text = "\n".join(extracted_lines)
                        print(f"✅ استخراج منظم ناجح للصفحة {page_num + 1}")
                        
                except Exception as e:
                    print(f"خطأ في الاستخراج المنظم للصفحة {page_num + 1}: {e}")
            
            # Method 3: Enhanced OCR for scanned pages
            if not text or len(text.strip()) < 50:
                try:
                    print(f"🔍 تطبيق OCR محسن على الصفحة {page_num + 1}")
                    
                    # High-resolution rendering for OCR
                    mat = fitz.Matrix(2.5, 2.5)  # Higher zoom for legal documents
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Apply enhanced OCR
                    ocr_text = enhanced_ocr_for_arabic(img_data)
                    
                    if ocr_text and len(ocr_text.strip()) > 30:
                        text = ocr_text
                        print(f"✅ OCR محسن نجح للصفحة {page_num + 1}: {len(text)} حرف")
                    else:
                        text = f"[صفحة ممسوحة ضوئياً {page_num+1} - تتطلب معالجة إضافية]"
                        print(f"⚠️ فشل OCR للصفحة {page_num + 1}")
                        
                except Exception as e:
                    print(f"خطأ في OCR للصفحة {page_num + 1}: {e}")
                    text = f"[خطأ في معالجة الصفحة {page_num+1}: {str(e)[:50]}]"
            
            # Clean and add to document
            if text.strip():
                cleaned_text = advanced_arabic_text_cleaner(text)
                if cleaned_text and len(cleaned_text.strip()) > 20:
                    full_text += cleaned_text + "\n\n"
        
        doc.close()
        
        # Final document-level cleaning
        final_text = advanced_arabic_text_cleaner(full_text)
        
        # Quality assessment
        if final_text:
            sample = final_text[:300] + "..." if len(final_text) > 300 else final_text
            print(f"📝 نموذج النص المستخرج من الوثيقة القانونية:")
            print(f"   {sample}")
            print(f"   الطول الإجمالي: {len(final_text)} حرف")
        else:
            print("❌ لم يتم استخراج أي نص مفيد من الوثيقة")
        
        return final_text
    
    except Exception as e:
        error_msg = f"خطأ في معالجة الوثيقة القانونية {pdf_path}: {e}"
        print(f"❌ {error_msg}")
        return error_msg

def smart_chunk_legal_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Smart chunking for Arabic legal documents"""
    if not text or len(text.strip()) < 50:
        return []
    
    chunks = []
    
    # Try to split by legal sections first
    legal_sections = re.split(r'(?=المادة\s+\d+|الفقرة\s+\d+|البند\s+\d+|الفصل\s+\d+)', text)
    
    if len(legal_sections) > 1:
        # Process each legal section
        for section in legal_sections:
            section = section.strip()
            if len(section) > 100:
                if len(section) <= chunk_size:
                    chunks.append(section)
                else:
                    # Further split large sections
                    chunks.extend(_split_large_section(section, chunk_size, overlap))
    else:
        # Fallback to sentence-based chunking
        chunks = _split_large_section(text, chunk_size, overlap)
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def _split_large_section(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Helper function to split large text sections"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunk = text[start:].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            break
        
        chunk = text[start:end]
        
        # Try to end at natural Arabic boundaries
        arabic_boundaries = ['۔', '؟', '!', '؛', '،', '.', '\n', ':', ')', '}']
        for boundary in arabic_boundaries:
            pos = chunk.rfind(boundary)
            if pos > chunk_size // 2:
                end = start + pos + 1
                chunk = text[start:end]
                break
        
        chunk = chunk.strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def create_enhanced_legal_prompt(query: str, context: str) -> str:
    """Create simple and effective Arabic-only prompt for legal questions"""
    
    prompt = f"""أنت خبير قانوني مصري متخصص في القانون المصري. يجب أن تجيب باللغة العربية فقط ولا تستخدم أي لغة أجنبية.

السؤال: {query}

النصوص القانونية المرجعية:
{context}

تعليمات:
1. أجب باللغة العربية الفصحى فقط
2. اعتمد على النصوص المرفقة أعلاه
3. اذكر رقم القانون والمادة إذا وجدت
4.you must answer in arabic only and you must give an answer

الإجابة:"""

    return prompt

def create_orchestrator_prompt(query: str, context: str, agent_responses: List[AgentResponse]) -> str:
    """Create enhanced Arabic orchestrator prompt that synthesizes multiple AI responses"""
    
    agent_answers = ""
    for i, agent_resp in enumerate(agent_responses, 1):
        agent_answers += f"""
📋 تحليل النموذج {i} ({agent_resp.model_name}):
{agent_resp.response}

{'='*80}
"""
    
    prompt = f"""أنت خبير قانوني مصري متخصص في القانون المصري والتشريعات العربية، تتمتع بخبرة واسعة في تحليل النصوص القانونية ودمج الآراء المتعددة. لديك القدرة على التفكير التحليلي العميق والاستنتاج المنطقي.

⚠️ تحذير هام: يجب أن تكتب باللغة العربية الفصحى فقط. لا تستخدم أي كلمات صينية أو إنجليزية أو أي لغة أخرى. استخدم المصطلحات القانونية العربية الأصيلة فقط.

<think>
سأحل هذا السؤال القانوني بعناية فائقة. أولاً، سأقرأ جميع المعلومات والوثائق المقدمة بدقة لفهم الموضوع القانوني المطروح.

سأحلل الإجابات المقدمة من النماذج المختلفة وأستخرج النقاط المهمة:
- سأراجع كل إجابة على حدة وأحدد نقاط القوة والضعف
- سأتحقق من دقة المراجع القانونية المذكورة (أرقام القوانين والمواد)
- سأقارن بين الإجابات لتحديد نقاط الاتفاق والاختلاف
- سأعتمد على النصوص القانونية المرفقة للتأكد من صحة المعلومات

بناءً على النصوص القانونية المرجعية، سأقوم بما يلي:
1. تحديد القوانين والمواد ذات الصلة بالسؤال
2. تحليل النصوص وفهم المعنى القانوني الدقيق
3. ربط النصوص بالسؤال المطروح
4. استخلاص الأحكام القانونية الواضحة

سأتأكد من دقة المراجع القانونية وأتحقق من:
- أرقام القوانين وسنوات الإصدار
- أرقام المواد والفقرات
- التعديلات والقوانين المكملة
- النطاق الزمني لتطبيق كل قانون

أخيراً، سأدمج هذه المعلومات في إجابة شاملة وواضحة باللغة العربية الفصحى، مع التأكد من تنظيم المعلومات بطريقة منطقية ومفهومة، وسأتجنب استخدام أي كلمات أجنبية أو صينية.
</think>

المطلوب منك:
1. **التحليل العميق**: ادرس الإجابات المقدمة بعناية فائقة واستخرج الحقائق القانونية الدقيقة
2. **التحقق من المراجع**: تأكد من صحة أرقام القوانين والمواد المذكورة
3. **المقارنة والتمييز**: قارن بين الإجابات وميز بين المعلومات الصحيحة والخاطئة
4. **التركيب المنطقي**: ادمج المعلومات الصحيحة في إجابة متماسكة ومنطقية
5. **الشمولية**: تأكد من تغطية جميع جوانب السؤال بوضوح

السؤال القانوني: {query}

النصوص القانونية المرجعية:
{context}

التحليلات المقدمة من النماذج المختلفة:
{agent_answers}

معايير الإجابة النهائية الإلزامية:
🚫 ممنوع منعاً باتاً استخدام أي كلمات صينية أو إنجليزية أو أجنبية
✅ يجب أن تكون الإجابة باللغة العربية الفصحى حصرياً ولا شيء غيرها
✅ استخدم المصطلحات القانونية العربية الأصيلة فقط
✅ اعتمد على النصوص القانونية المرفقة كمصدر أساسي
✅ اذكر أرقام القوانين والمواد بدقة تامة باللغة العربية
✅ نظم الإجابة بعناوين فرعية واضحة عند الحاجة
✅ في حالة وجود تناقضات، وضح الرأي الأصح مع التبرير القانوني
✅ قدم إجابة عملية ومفيدة للمستفيد
✅ تجنب ذكر أنك تعتمد على إجابات متعددة، واجعل الإجابة تبدو كتحليل موحد
✅ تأكد من عدم وجود أي أحرف أو كلمات غير عربية في إجابتك

الإجابة القانونية النهائية المدمجة باللغة العربية الفصحى فقط:"""

    return prompt

def query_single_agent(prompt: str, model_name: str) -> AgentResponse:
    """Query a single agent model and return structured response"""
    start_time = time.time()
    
    try:
        print(f"🔄 استدعاء النموذج {model_name}...")
        cmd = ["ollama", "run", model_name]
        process = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt, timeout=300)  # 5 minute timeout per agent
        
        processing_time = time.time() - start_time
        
        if process.returncode != 0:
            print(f"❌ خطأ في النموذج {model_name}: {stderr}")
            return AgentResponse(
                model_name=model_name,
                response=f"❌ خطأ في النموذج: {stderr}",
                processing_time=processing_time,
                error=stderr
            )
        
        print(f"✅ تم التحليل بنجاح من {model_name} ({len(stdout)} حرف) في {processing_time:.2f}s")
        # Clean the response to remove any foreign text
        cleaned_response = clean_arabic_response(stdout.strip())
        return AgentResponse(
            model_name=model_name,
            response=cleaned_response,
            processing_time=processing_time
        )
        
    except subprocess.TimeoutExpired:
        processing_time = time.time() - start_time
        print(f"❌ انتهت مهلة الاستجابة للنموذج {model_name}")
        return AgentResponse(
            model_name=model_name,
            response="❌ انتهت مهلة الاستجابة",
            processing_time=processing_time,
            error="timeout"
        )
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ خطأ في النموذج {model_name}: {str(e)}")
        return AgentResponse(
            model_name=model_name,
            response=f"❌ خطأ في التحليل: {str(e)}",
            processing_time=processing_time,
            error=str(e)
        )

def clean_arabic_response(text: str) -> str:
    """Clean response to ensure it's purely Arabic"""
    import re
    
    try:
        # Remove Chinese characters (CJK Unified Ideographs)
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        
        # Remove common English/foreign words that shouldn't be in Arabic legal text
        foreign_words = [
            # English words
            r'\bLIABLE\b', r'\bVAT\b', r'\bwithholding\b', r'\btax\b', 
            r'\bvalue\b', r'\bcorp\b', r'\binc\b', r'\bltd\b',
            r'\bcompany\b', r'\bincome\b', r'\bprofit\b', r'\brate\b',
            r'\blegal\b', r'\banalysis\b', r'\barticle\b', r'\blaw\b',
            r'\bregulation\b', r'\bgoverns\b', r'\bshall\b', r'\bsubject\b',
            # Mixed Arabic-foreign constructions
            r'ال規定', r'净利润', r'与此', r'規定', r'النظام', 
            # Chinese characters commonly mixed in
            r'净', r'利', r'润', r'規', r'定', r'条', r'例',
            r'法', r'律', r'税', r'务', r'公', r'司'
        ]
        
        for pattern in foreign_words:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Additional specific Chinese character ranges
        text = re.sub(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+', '', text)
        
        # Remove Latin characters that don't belong in Arabic legal text
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # Clean up mixed punctuation that might come from foreign text
        text = re.sub(r'[''""„""‚''``''´´""]+', '', text)
        
        # Remove any remaining non-Arabic, non-number, non-punctuation characters
        # Keep Arabic, numbers, punctuation, and whitespace
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0660-\u0669\u06F0-\u06F9\u0020-\u007F\u00A0-\u00FF\n\r\t .,;:!?()[\]{}\-/°%#*]+', ' ', text)
        
        # Clean up multiple spaces and extra newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove standalone bullet points or numbering that might be left over
        text = re.sub(r'\n\s*[\-\*\•]\s*\n', '\n', text)
        text = re.sub(r'\n\s*\d+\.\s*\n', '\n', text)
        
        return text.strip()
    except Exception as e:
        print(f"خطأ في تنظيف النص: {e}")
        return text

def extract_answer_after_think(response: str) -> str:
    """Extract the answer portion after </think> tag and clean it"""
    try:
        # Find the </think> tag and extract everything after it
        think_end = response.find('</think>')
        if think_end != -1:
            # Extract content after </think> and clean it up
            answer = response[think_end + len('</think>'):].strip()
            # Remove any remaining whitespace and newlines at the beginning
            answer = '\n'.join(line.strip() for line in answer.split('\n') if line.strip())
            
            # Clean the answer to ensure it's purely Arabic
            answer = clean_arabic_response(answer)
            
            return answer if answer else response
        else:
            # If no </think> tag found, clean the original response
            return clean_arabic_response(response)
    except Exception as e:
        print(f"خطأ في استخراج الإجابة: {e}")
        return clean_arabic_response(response)

def run_multi_agent_analysis(query: str, context: str) -> tuple[List[AgentResponse], str]:
    """Run multi-agent analysis with 2 models sequentially, then orchestrate"""
    print("🚀 بدء التحليل متعدد النماذج...")
    
    # Step 1: Run 2 agent models sequentially
    agent_responses = []
    prompt = create_enhanced_legal_prompt(query, context)
    
    for model_name in AGENT_MODELS:
        print(f"📡 تشغيل النموذج {model_name}...")
        agent_response = query_single_agent(prompt, model_name)
        agent_responses.append(agent_response)
        print(f"✅ انتهى النموذج {model_name}")
    
    print("🔄 تحليل الإجابات وتركيبها...")
    
    # Step 2: Orchestrate responses using deepseek
    orchestrator_prompt = create_orchestrator_prompt(query, context, agent_responses)
    raw_orchestrator_response = query_ollama_enhanced(orchestrator_prompt, ORCHESTRATOR_MODEL)
    
    # Extract only the answer after </think>
    orchestrator_response = extract_answer_after_think(raw_orchestrator_response)
    
    print("✅ تم انهاء التحليل متعدد النماذج")
    return agent_responses, orchestrator_response

def query_ollama_enhanced(prompt: str, model_name: str = "qwen2.5:14b") -> str:
    """Enhanced Ollama query for legal responses"""
    try:
        print(f"🔄 استدعاء Ollama للتحليل القانوني بنموذج: {model_name}")
        cmd = ["ollama", "run", model_name]
        process = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt, timeout=420)  # Longer timeout for complex legal analysis
        
        if process.returncode != 0:
            print(f"❌ خطأ Ollama: return_code={process.returncode}, stderr={stderr}")
            return f"❌ خطأ في النظام: {stderr}"
        
        print(f"✅ تم التحليل القانوني بنجاح - عدد الأحرف: {len(stdout)}")
        # Clean the response to ensure it's purely Arabic
        cleaned_response = clean_arabic_response(stdout.strip())
        return cleaned_response
        
    except subprocess.TimeoutExpired:
        print("❌ انتهت مهلة الاستجابة")
        return "❌ انتهت مهلة الاستجابة - السؤال يتطلب وقتاً أطول للتحليل"
    except FileNotFoundError:
        print("❌ Ollama غير موجود")
        return "❌ نظام الذكاء الاصطناعي غير متاح حالياً"
    except Exception as e:
        print(f"❌ خطأ في query_ollama: {str(e)}")
        return f"❌ خطأ في التحليل: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced legal document system"""
    global embedding_model, faiss_index, documents
    
    print("🚀 تحميل نظام تحليل الوثائق القانونية المحسن...")
    
    # Initialize OCR
    ocr_success = initialize_ocr()
    
    # Initialize embedding model
    embedding_success = initialize_embedding_model()
    
    if not embedding_success:
        print("❌ فشل تحميل نموذج التشفير")
        return
    
    # Try to load existing embeddings first
    if load_embeddings_and_documents():
        print("🎉 تم تحميل نظام الوثائق القانونية من الملف المحفوظ بنجاح!")
        return
    
    # If no cached embeddings, process PDFs
    print("📚 معالجة الوثائق القانونية...")
    documents = []
    
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ مجلد الوثائق القانونية غير موجود: {PDF_FOLDER}")
        return
    
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    total_files = len(pdf_files)
    print(f"📄 وجدت {total_files} وثيقة قانونية")
    
    for file_index, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"📖 معالجة الوثيقة ({file_index}/{total_files}): {filename}")
        
        text = extract_text_from_legal_pdf(pdf_path)
        
        if text and len(text.strip()) > 100:
            chunks = smart_chunk_legal_text(text)
            if chunks:
                for j, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': filename,
                        'chunk_id': j,
                        'metadata': f"{filename} - القسم {j+1}"
                    })
                print(f"✅ تم إنشاء {len(chunks)} قسم قانوني")
            else:
                documents.append({
                    'content': text.strip(),
                    'source': filename,
                    'chunk_id': 0,
                    'metadata': f"{filename} - النص الكامل"
                })
                print(f"✅ تم حفظ الوثيقة كقسم واحد")
        else:
            print(f"⚠️ لا يوجد محتوى قانوني كافٍ في {filename}")
    
    print(f"📊 إجمالي الأقسام القانونية: {len(documents)}")
    
    # Create embeddings and FAISS index
    if documents and embedding_model:
        print("🔍 إنشاء فهرس البحث المتقدم...")
        
        # Generate embeddings
        document_texts = [doc['content'] for doc in documents]
        try:
            document_embeddings = embedding_model.encode(document_texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = document_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize and add to index
            faiss.normalize_L2(document_embeddings.astype('float32'))
            faiss_index.add(document_embeddings.astype('float32'))
            
            print(f"✅ تم إنشاء فهرس FAISS بـ {faiss_index.ntotal} تشفير")
            
            # Save embeddings for future use
            save_embeddings_and_documents(document_embeddings)
            
        except Exception as e:
            print(f"❌ خطأ في إنشاء فهرس البحث: {e}")
    
    print("🎉 تم تحميل نظام الوثائق القانونية بنجاح!")

def search_legal_documents(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Enhanced search for legal documents"""
    if faiss_index is None or not documents or embedding_model is None:
        return []
    
    try:
        # Encode query
        query_embedding = embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = faiss_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(documents):
                # Fix similarity score - ensure it's between 0 and 1
                # FAISS cosine similarity can sometimes exceed 1.0 due to numerical precision
                normalized_score = max(0.0, min(1.0, float(score)))
                
                results.append({
                    'content': documents[idx]['content'],
                    'source': documents[idx]['source'],
                    'metadata': documents[idx]['metadata'],
                    'similarity_score': normalized_score,
                    'chunk_id': documents[idx]['chunk_id']
                })
        
        return results
    except Exception as e:
        print(f"خطأ في البحث: {e}")
        return []

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_loaded": len(documents),
        "embedding_model_loaded": embedding_model is not None,
        "faiss_index_ready": faiss_index is not None,
        "ocr_ready": easyocr_reader is not None,
        "system_type": "Enhanced Arabic Legal Documents RAG"
    }

@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    return SystemStats(
        totalDocuments=len(set(doc['source'] for doc in documents)),
        totalChunks=len(documents),
        indexSize=len(documents),
        lastUpdated="2024-01-01",
        processingStatus="ready",
        agentModels=AGENT_MODELS,
        orchestratorModel=ORCHESTRATOR_MODEL
    )

@app.post("/api/search", response_model=MultiAgentRAGResponse)
async def search_legal_documents_api(request: SearchRequest):
    try:
        start_time = asyncio.get_event_loop().time()
        
        search_results = search_legal_documents(request.query, request.top_k or TOP_K)
        
        if not search_results:
            return MultiAgentRAGResponse(
                question=request.query,
                retrieved_documents=[],
                agent_responses=[],
                orchestrator_response="لم يتم العثور على نصوص قانونية ذات صلة بالسؤال المطروح في قاعدة البيانات القانونية المتاحة",
                final_answer="",
                total_processing_time=asyncio.get_event_loop().time() - start_time
            )
        
        # Create enhanced legal context
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            content = result['content'][:800] + "..." if len(result['content']) > 800 else result['content']
            context_parts.append(f"📄 الوثيقة القانونية {i} - {result['source']}:\n{content}")
        
        context = "\n" + "="*80 + "\n".join(context_parts) + "\n" + "="*80
        
        # Run multi-agent analysis
        agent_responses, orchestrator_answer = run_multi_agent_analysis(request.query, context)
        
        # Prepare response
        search_result_objects = [
            SearchResult(
                content=result['content'],
                source=result['source'],
                metadata=result['metadata'],
                similarity_score=result['similarity_score'],
                chunk_id=result['chunk_id']
            )
            for result in search_results
        ]
        
        final_answer = orchestrator_answer
        
        return MultiAgentRAGResponse(
            question=request.query,
            retrieved_documents=search_result_objects,
            agent_responses=agent_responses,
            orchestrator_response=orchestrator_answer,
            final_answer=final_answer,
            total_processing_time=asyncio.get_event_loop().time() - start_time
        )
        
    except Exception as e:
        return MultiAgentRAGResponse(
            question=request.query,
            retrieved_documents=[],
            agent_responses=[],
            orchestrator_response="",
            final_answer="",
            error=str(e)
        )

@app.post("/api/search/stream", response_model=MultiAgentRAGResponse)
async def search_legal_documents_stream(request: SearchRequest):
    """Enhanced legal document search with streaming-like response"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        if not request.query.strip():
            return MultiAgentRAGResponse(
                question=request.query,
                retrieved_documents=[],
                agent_responses=[],
                orchestrator_response="",
                final_answer="",
                error="السؤال لا يمكن أن يكون فارغاً"
            )
        
        print(f"🔍 البحث في الوثائق القانونية عن: {request.query}")
        
        # Search for relevant legal documents
        search_results = search_legal_documents(request.query, request.top_k or TOP_K)
        
        print(f"📊 نتائج البحث القانوني: {len(search_results)} وثيقة")
        for i, result in enumerate(search_results[:3]):
            print(f"📄 وثيقة {i+1}: {result['source']} - نتيجة: {result['similarity_score']:.4f}")
            sample = result['content'][:100].replace('\n', ' ') + "..."
            print(f"   📝 المحتوى: {sample}")
        
        if not search_results:
            print("❌ لم يتم العثور على وثائق قانونية ذات صلة")
            return MultiAgentRAGResponse(
                question=request.query,
                retrieved_documents=[],
                agent_responses=[],
                orchestrator_response="لم يتم العثور على نصوص قانونية مطابقة للسؤال في قاعدة البيانات القانونية المصرية المتاحة. يُرجى إعادة صياغة السؤال أو التأكد من أن الموضوع مشمول في القوانين المتاحة.",
                final_answer="",
                total_processing_time=asyncio.get_event_loop().time() - start_time
            )
        
        print(f"📚 تم العثور على {len(search_results)} وثيقة قانونية ذات صلة")
        
        # Create comprehensive legal context
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            content = result['content'][:900] + "..." if len(result['content']) > 900 else result['content']
            context_parts.append(f"📖 المرجع القانوني {i} من الوثيقة [{result['source']}]:\n{content}")
        
        context = "\n" + "🔷"*50 + "\n".join(context_parts) + "\n" + "🔷"*50
        
        print("⚖️ معالجة السؤال القانوني مع الذكاء الاصطناعي المتخصص...")
        
        # Run multi-agent analysis
        agent_responses, orchestrator_answer = run_multi_agent_analysis(request.query, context)
        
        final_answer = orchestrator_answer
        
        # Prepare response objects
        search_result_objects = [
            SearchResult(
                content=result['content'],
                source=result['source'],
                metadata=result['metadata'],
                similarity_score=result['similarity_score'],
                chunk_id=result['chunk_id']
            )
            for result in search_results
        ]
        
        return MultiAgentRAGResponse(
            question=request.query,
            retrieved_documents=search_result_objects,
            agent_responses=agent_responses,
            orchestrator_response=orchestrator_answer,
            final_answer=final_answer,
            total_processing_time=asyncio.get_event_loop().time() - start_time
        )
        
    except Exception as e:
        print(f"❌ خطأ في معالجة السؤال القانوني: {str(e)}")
        return MultiAgentRAGResponse(
            question=request.query,
            retrieved_documents=[],
            agent_responses=[],
            orchestrator_response="",
            final_answer="",
            error=f"خطأ في النظام: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 