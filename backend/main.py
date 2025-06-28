from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import glob
import fitz  # PyMuPDF
import numpy as np
import faiss
import re
import json
from typing import List, Dict, Optional
import asyncio
import warnings
import unicodedata
import io
import cv2
from PIL import Image
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display
import pickle
import time
from groq import Groq
from langdetect import detect
from deep_translator import GoogleTranslator
from textblob import TextBlob
from spellchecker import SpellChecker

# Sentence embeddings for better search  
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

app = FastAPI(title="Arabic Legal RAG System with Agentic Workflow")

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

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
GROQ_MODEL = "meta-llama/llama-3.1-70b-versatile"

# Global variables
embedding_model = None
faiss_index = None
documents = []
groq_client = None
spell_checker = None

class ChatMessage(BaseModel):
    message: str

class AgentResponse(BaseModel):
    agent_name: str
    response: str
    processing_time: float
    error: Optional[str] = None

class SearchItem(BaseModel):
    query: str
    priority: int
    category: str

class RAGResult(BaseModel):
    search_item: str
    documents: List[Dict]
    summary: str

class AgentWorkflowResponse(BaseModel):
    collected_info: Dict[str, str]
    search_items: List[SearchItem]
    rag_results: List[RAGResult]
    final_recommendation: str
    processing_time: float
    error: Optional[str] = None

def initialize_groq():
    """Initialize Groq client"""
    global groq_client
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ تم تحميل Groq بنجاح")
        return True
    except Exception as e:
        print(f"❌ فشل تحميل Groq: {e}")
        return False

def initialize_spell_checker():
    """Initialize spell checker for text correction"""
    global spell_checker
    try:
        spell_checker = SpellChecker(language='en')
        print("✅ تم تحميل مصحح الإملاء بنجاح")
        return True
    except Exception as e:
        print(f"❌ فشل تحميل مصحح الإملاء: {e}")
        return False

def correct_text(text: str) -> str:
    """Auto-correct text using spell checker"""
    if not spell_checker:
        return text
    
    try:
        # Split text into words
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Clean word from punctuation for checking
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word and clean_word in spell_checker:
                corrected_words.append(word)
            elif clean_word:
                # Get correction
                correction = spell_checker.correction(clean_word)
                if correction and correction != clean_word:
                    # Replace the clean part while keeping punctuation
                    corrected_word = word.replace(clean_word, correction)
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    except Exception as e:
        print(f"خطأ في تصحيح النص: {e}")
        return text

def detect_and_translate_to_arabic(text: str) -> str:
    """Detect language and translate to Arabic if needed"""
    try:
        # Detect language
        detected_lang = detect(text)
        print(f"🔍 اللغة المكتشفة: {detected_lang}")
        
        # If already Arabic, return as is
        if detected_lang == 'ar':
            return text
        
        # Translate to Arabic
        translator = GoogleTranslator(source=detected_lang, target='ar')
        
        # Split text into chunks for translation (Google Translator has limits)
        max_chunk_size = 4500
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        translated_chunks = []
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
                time.sleep(0.1)  # Small delay to avoid rate limiting
            except Exception as e:
                print(f"خطأ في ترجمة جزء من النص: {e}")
                translated_chunks.append(chunk)  # Keep original if translation fails
        
        translated_text = ' '.join(translated_chunks)
        print(f"✅ تم ترجمة النص من {detected_lang} إلى العربية")
        return translated_text
        
    except Exception as e:
        print(f"خطأ في الكشف عن اللغة أو الترجمة: {e}")
        return text

def extract_text_with_tesseract(pdf_path: str) -> str:
    """Extract text from PDF using Tesseract OCR with auto-correction and translation"""
    try:
        print(f"📄 معالجة الوثيقة باستخدام Tesseract: {os.path.basename(pdf_path)}")
        
        doc = fitz.Document(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # High resolution for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            img_array = np.array(image)
            
            # Preprocess image for better OCR
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply image processing for better OCR
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Use Tesseract to extract text
            try:
                # Try multiple configurations
                configs = [
                    '--oem 3 --psm 6',  # Standard
                    '--oem 3 --psm 4',  # Single column
                    '--oem 3 --psm 3',  # Fully automatic
                ]
                
                page_text = ""
                for config in configs:
                    try:
                        extracted_text = pytesseract.image_to_string(denoised, config=config)
                        if extracted_text.strip() and len(extracted_text.strip()) > 20:
                            page_text = extracted_text
                            break
                    except:
                        continue
                
                if page_text:
                    # Step 1: Auto-correct the text
                    corrected_text = correct_text(page_text)
                    
                    # Step 2: Detect language and translate to Arabic if needed
                    arabic_text = detect_and_translate_to_arabic(corrected_text)
                    
                    full_text += arabic_text + "\n\n"
                    print(f"✅ تم معالجة الصفحة {page_num + 1}")
                else:
                    print(f"⚠️ لم يتم استخراج نص من الصفحة {page_num + 1}")
                    
            except Exception as e:
                print(f"خطأ في OCR للصفحة {page_num + 1}: {e}")
        
        doc.close()
        
        # Final cleaning
        final_text = clean_arabic_text(full_text)
        
        if final_text:
            print(f"📝 تم استخراج وترجمة {len(final_text)} حرف من الوثيقة")
        else:
            print("❌ لم يتم استخراج أي نص مفيد من الوثيقة")
        
        return final_text
        
    except Exception as e:
        error_msg = f"خطأ في معالجة الوثيقة {pdf_path}: {e}"
        print(f"❌ {error_msg}")
        return error_msg

def clean_arabic_text(text: str) -> str:
    """Clean and normalize Arabic text"""
    if not text or not text.strip():
        return ""
    
    try:
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove unwanted characters but keep Arabic, numbers, and punctuation
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0660-\u0669\u06F0-\u06F9\u0020-\u007F\u00A0-\u00FF\n\r\t .,;:!?\-/°%()]+', ' ', text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
        
    except Exception as e:
        print(f"خطأ في تنظيف النص العربي: {e}")
        return text.strip() if text else ""

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
        
        save_data = {
            'embeddings': embeddings_array,
            'documents': documents,
            'index_size': len(embeddings_array)
        }
        
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ تم حفظ {len(documents)} وثيقة و {len(embeddings_array)} تضمين")
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
        
        with open(EMBEDDINGS_FILE, 'rb') as f:
            save_data = pickle.load(f)
        
        documents = save_data['documents']
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

def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Smart chunking for text"""
    if not text or len(text.strip()) < 50:
        return []
    
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
        
        # Try to end at natural boundaries
        boundaries = ['.', '!', '?', '\n', ':', ';']
        for boundary in boundaries:
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

async def query_groq(prompt: str, system_prompt: str = None) -> str:
    """Query Groq API"""
    if not groq_client:
        return "❌ Groq client not initialized"
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"خطأ في استدعاء Groq: {e}")
        return f"❌ خطأ في النظام: {str(e)}"

def search_documents(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Search for relevant documents"""
    if faiss_index is None or not documents or embedding_model is None:
        return []
    
    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = faiss_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(documents):
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

async def information_collection_agent(user_message: str) -> Dict[str, str]:
    """Agent to collect business information from user"""
    
    system_prompt = """أنت محامي خبير في القانون المصري متخصص في تسجيل الشركات والامتثال الضريبي. 
مهمتك هي جمع المعلومات الأساسية من العميل لضمان التسجيل الصحيح والامتثال للقوانين.

يجب أن تسأل عن:
1. هيكل الشركة: نوع الكيان القانوني (فردية، ذات مسؤولية محدودة، مساهمة)
2. المعلومات المالية: الإيرادات المتوقعة، رأس المال، الالتزامات الضريبية السابقة
3. الأنشطة التجارية: الأنشطة الرئيسية، نوع الخدمات/المنتجات، العمليات الدولية
4. التاريخ الضريبي: التسجيل السابق، الالتزامات المعلقة
5. الامتثال والوثائق: الوثائق المطلوبة، المواعيد النهائية

اجمع المعلومات بطريقة ودية ومهنية واطرح أسئلة متابعة حسب الحاجة."""

    prompt = f"""بناءً على رسالة العميل التالية، اجمع المعلومات المطلوبة:

رسالة العميل: {user_message}

قم بتحليل الرسالة واستخراج أي معلومات متوفرة، ثم اطرح أسئلة محددة للحصول على المعلومات المفقودة.

قدم الرد في شكل JSON مع المفاتيح التالية:
- collected_info: معلومات تم جمعها من الرسالة
- follow_up_questions: أسئلة للحصول على معلومات إضافية
- next_steps: الخطوات التالية المقترحة"""

    response = await query_groq(prompt, system_prompt)
    
    try:
        # Try to parse JSON response
        import json
        parsed_response = json.loads(response)
        return parsed_response
    except:
        # If JSON parsing fails, return structured response
        return {
            "collected_info": {"general_inquiry": user_message},
            "follow_up_questions": response,
            "next_steps": "جمع المزيد من المعلومات"
        }

async def search_planning_agent(collected_info: Dict) -> List[SearchItem]:
    """Agent to plan search queries based on collected information"""
    
    system_prompt = """أنت خبير في البحث القانوني. مهمتك هي تحديد النقاط الرئيسية للبحث في قاعدة البيانات القانونية 
بناءً على المعلومات المجمعة من العميل.

قم بإنشاء قائمة من عناصر البحث المحددة والقابلة للبحث التي ستساعد في العثور على القوانين واللوائح ذات الصلة."""

    info_text = json.dumps(collected_info, ensure_ascii=False, indent=2)
    
    prompt = f"""بناءً على المعلومات التالية المجمعة من العميل:

{info_text}

قم بإنشاء قائمة من 5-8 عناصر بحث محددة للعثور على القوانين واللوائح ذات الصلة.

قدم الرد في شكل JSON مع قائمة من العناصر، كل عنصر يحتوي على:
- query: نص البحث
- priority: الأولوية (1-5)
- category: فئة القانون (ضريبي، تجاري، عمالي، إلخ)

مثال:
[
  {{"query": "تسجيل شركة ذات مسؤولية محدودة", "priority": 5, "category": "تجاري"}},
  {{"query": "الضرائب على الشركات الصغيرة", "priority": 4, "category": "ضريبي"}}
]"""

    response = await query_groq(prompt, system_prompt)
    
    try:
        parsed_response = json.loads(response)
        search_items = []
        for item in parsed_response:
            search_items.append(SearchItem(
                query=item.get('query', ''),
                priority=item.get('priority', 3),
                category=item.get('category', 'عام')
            ))
        return search_items
    except:
        # Fallback search items
        return [
            SearchItem(query="تسجيل الشركات", priority=5, category="تجاري"),
            SearchItem(query="الضرائب على الشركات", priority=4, category="ضريبي"),
            SearchItem(query="الامتثال القانوني", priority=3, category="عام")
        ]

async def rag_search_agent(search_items: List[SearchItem]) -> List[RAGResult]:
    """Agent to perform RAG searches and summarize results"""
    
    rag_results = []
    
    for item in search_items:
        # Search for documents
        documents = search_documents(item.query, top_k=3)
        
        if documents:
            # Create context from documents
            context = "\n\n".join([doc['content'][:500] + "..." for doc in documents])
            
            # Summarize findings
            system_prompt = """أنت خبير قانوني. قم بتلخيص النتائج القانونية بطريقة واضحة ومفيدة للعميل."""
            
            prompt = f"""بناءً على البحث عن "{item.query}"، تم العثور على النصوص القانونية التالية:

{context}

قدم ملخصاً واضحاً ومفيداً للنقاط الرئيسية والمتطلبات القانونية ذات الصلة."""

            summary = await query_groq(prompt, system_prompt)
            
            rag_results.append(RAGResult(
                search_item=item.query,
                documents=documents,
                summary=summary
            ))
        else:
            rag_results.append(RAGResult(
                search_item=item.query,
                documents=[],
                summary="لم يتم العثور على نتائج ذات صلة في قاعدة البيانات."
            ))
    
    return rag_results

async def recommendation_agent(collected_info: Dict, rag_results: List[RAGResult]) -> str:
    """Agent to provide final recommendations"""
    
    system_prompt = """أنت محامي خبير في القانون المصري. قدم توصيات شاملة ومفصلة للعميل 
بناءً على المعلومات المجمعة ونتائج البحث القانوني."""

    info_text = json.dumps(collected_info, ensure_ascii=False, indent=2)
    results_text = "\n\n".join([f"البحث: {result.search_item}\nالملخص: {result.summary}" for result in rag_results])
    
    prompt = f"""بناءً على المعلومات التالية والبحث القانوني:

معلومات العميل:
{info_text}

نتائج البحث القانوني:
{results_text}

قدم توصيات شاملة تتضمن:
1. الخطوات المطلوبة للتسجيل
2. المتطلبات القانونية والضريبية
3. الوثائق المطلوبة
4. المواعيد النهائية المهمة
5. التحذيرات والنصائح المهمة

قدم الرد بطريقة منظمة وواضحة."""

    recommendation = await query_groq(prompt, system_prompt)
    return recommendation

@app.on_event("startup")
async def startup_event():
    """Initialize the system"""
    global embedding_model, faiss_index, documents
    
    print("🚀 تحميل نظام RAG القانوني مع سير العمل الذكي...")
    
    # Initialize components
    groq_success = initialize_groq()
    spell_success = initialize_spell_checker()
    embedding_success = initialize_embedding_model()
    
    if not embedding_success:
        print("❌ فشل تحميل نموذج التشفير")
        return
    
    # Try to load existing embeddings
    if load_embeddings_and_documents():
        print("🎉 تم تحميل النظام من الملف المحفوظ بنجاح!")
        return
    
    # Process PDFs if no cached embeddings
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
        
        # Use Tesseract with auto-correction and translation
        text = extract_text_with_tesseract(pdf_path)
        
        if text and len(text.strip()) > 100:
            chunks = smart_chunk_text(text)
            if chunks:
                for j, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': filename,
                        'chunk_id': j,
                        'metadata': f"{filename} - القسم {j+1}"
                    })
                print(f"✅ تم إنشاء {len(chunks)} قسم")
            else:
                documents.append({
                    'content': text.strip(),
                    'source': filename,
                    'chunk_id': 0,
                    'metadata': f"{filename} - النص الكامل"
                })
                print(f"✅ تم حفظ الوثيقة كقسم واحد")
        else:
            print(f"⚠️ لا يوجد محتوى كافٍ في {filename}")
    
    print(f"📊 إجمالي الأقسام: {len(documents)}")
    
    # Create embeddings and FAISS index
    if documents and embedding_model:
        print("🔍 إنشاء فهرس البحث...")
        
        document_texts = [doc['content'] for doc in documents]
        try:
            document_embeddings = embedding_model.encode(document_texts, show_progress_bar=True)
            
            dimension = document_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(document_embeddings.astype('float32'))
            faiss_index.add(document_embeddings.astype('float32'))
            
            print(f"✅ تم إنشاء فهرس FAISS بـ {faiss_index.ntotal} تشفير")
            
            save_embeddings_and_documents(document_embeddings)
            
        except Exception as e:
            print(f"❌ خطأ في إنشاء فهرس البحث: {e}")
    
    print("🎉 تم تحميل النظام بنجاح!")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_loaded": len(documents),
        "embedding_model_loaded": embedding_model is not None,
        "faiss_index_ready": faiss_index is not None,
        "groq_ready": groq_client is not None,
        "system_type": "Arabic Legal RAG with Agentic Workflow"
    }

@app.post("/api/chat", response_model=AgentWorkflowResponse)
async def chat_with_agent(message: ChatMessage):
    """Main chat endpoint with agentic workflow"""
    try:
        start_time = time.time()
        
        print(f"💬 رسالة جديدة: {message.message}")
        
        # Step 1: Information Collection
        print("🔍 جمع المعلومات...")
        collected_info = await information_collection_agent(message.message)
        
        # Step 2: Search Planning
        print("📋 تخطيط البحث...")
        search_items = await search_planning_agent(collected_info)
        
        # Step 3: RAG Search
        print("🔎 البحث في قاعدة البيانات...")
        rag_results = await rag_search_agent(search_items)
        
        # Step 4: Final Recommendation
        print("📝 إعداد التوصيات...")
        final_recommendation = await recommendation_agent(collected_info, rag_results)
        
        processing_time = time.time() - start_time
        
        return AgentWorkflowResponse(
            collected_info=collected_info,
            search_items=search_items,
            rag_results=rag_results,
            final_recommendation=final_recommendation,
            processing_time=processing_time
        )
        
    except Exception as e:
        return AgentWorkflowResponse(
            collected_info={},
            search_items=[],
            rag_results=[],
            final_recommendation="",
            processing_time=0,
            error=str(e)
        )

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a new document"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        text = extract_text_with_tesseract(temp_path)
        
        if text and len(text.strip()) > 100:
            chunks = smart_chunk_text(text)
            new_documents = []
            
            for j, chunk in enumerate(chunks):
                new_doc = {
                    'content': chunk,
                    'source': file.filename,
                    'chunk_id': j,
                    'metadata': f"{file.filename} - القسم {j+1}"
                }
                documents.append(new_doc)
                new_documents.append(new_doc)
            
            # Update embeddings
            if embedding_model and faiss_index:
                new_texts = [doc['content'] for doc in new_documents]
                new_embeddings = embedding_model.encode(new_texts)
                faiss.normalize_L2(new_embeddings.astype('float32'))
                faiss_index.add(new_embeddings.astype('float32'))
                
                # Save updated embeddings
                all_embeddings = np.vstack([
                    np.array([embedding_model.encode([doc['content']]) for doc in documents[:-len(new_documents)]]).squeeze(),
                    new_embeddings
                ])
                save_embeddings_and_documents(all_embeddings)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return {
                "message": f"تم رفع ومعالجة الوثيقة بنجاح. تم إنشاء {len(chunks)} قسم نصي.",
                "chunks_created": len(chunks),
                "filename": file.filename
            }
        else:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Could not extract sufficient text from the document")
            
    except Exception as e:
        return {"error": f"خطأ في رفع الوثيقة: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)