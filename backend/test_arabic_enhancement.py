#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Enhanced Arabic Legal Document Processing
Tests the improved OCR and Arabic text cleaning functions
"""

import requests
import json
import time

# Test the enhanced backend
def test_enhanced_backend():
    """Test the enhanced Arabic legal document processing"""
    
    base_url = "http://localhost:8000"
    
    # Test health check
    print("🔍 اختبار حالة النظام المحسن...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ النظام يعمل بشكل صحيح:")
            print(f"   - الوثائق المحملة: {health.get('documents_loaded', 0)}")
            print(f"   - نموذج التشفير: {'✅' if health.get('embedding_model_loaded') else '❌'}")
            print(f"   - فهرس البحث: {'✅' if health.get('faiss_index_ready') else '❌'}")
            print(f"   - محرك OCR: {'✅' if health.get('ocr_ready') else '❌'}")
            print(f"   - نوع النظام: {health.get('system_type', 'غير محدد')}")
        else:
            print(f"❌ فشل في الاتصال بالنظام: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ خطأ في الاتصال: {e}")
        return False
    
    # Test legal questions
    legal_questions = [
        "ما هو القانون رقم ١٦٠ لسنة ٢٠٢٤؟",
        "ما هي أحكام تجديد العمل بالقانون؟",
        "ما هي إجراءات إنهاء النزاعات الضريبية؟",
        "ما هي المادة الأولى من القانون؟",
        "ما هي الجهة المختصة بالفصل في الطلبات؟"
    ]
    
    print("\n🎯 اختبار الأسئلة القانونية المحسنة...")
    
    for i, question in enumerate(legal_questions, 1):
        print(f"\n📋 السؤال {i}: {question}")
        
        try:
            # Test the enhanced search
            response = requests.post(
                f"{base_url}/api/search/stream",
                json={"query": question, "top_k": 3},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ تم الحصول على إجابة:")
                print(f"   - عدد الوثائق المسترجعة: {len(result.get('retrieved_documents', []))}")
                print(f"   - وقت المعالجة: {result.get('processing_time', 0):.2f} ثانية")
                
                # Display a sample of the Arabic answer
                answer = result.get('model_answer', '')
                if answer:
                    # Check if answer is in Arabic
                    arabic_chars = len([c for c in answer if '\u0600' <= c <= '\u06FF'])
                    total_chars = len(answer.replace(' ', ''))
                    arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
                    
                    print(f"   - طول الإجابة: {len(answer)} حرف")
                    print(f"   - نسبة الأحرف العربية: {arabic_ratio:.2%}")
                    
                    if arabic_ratio > 0.8:
                        print("   ✅ الإجابة باللغة العربية بشكل صحيح")
                    else:
                        print("   ⚠️ الإجابة تحتوي على نسبة قليلة من العربية")
                    
                    # Show preview
                    preview = answer[:200] + "..." if len(answer) > 200 else answer
                    print(f"   📝 معاينة الإجابة: {preview}")
                else:
                    print("   ⚠️ لم يتم الحصول على إجابة")
                
                # Show document sources
                docs = result.get('retrieved_documents', [])
                if docs:
                    print(f"   📚 الوثائق المرجعية:")
                    for j, doc in enumerate(docs[:2], 1):
                        score = doc.get('similarity_score', 0)
                        source = doc.get('source', 'غير محدد')
                        print(f"      {j}. {source} (نتيجة: {score:.3f})")
                
            else:
                print(f"   ❌ فشل في الاستعلام: {response.status_code}")
                if response.text:
                    print(f"      الخطأ: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ خطأ في الاستعلام: {e}")
        
        # Wait between requests
        time.sleep(1)
    
    print("\n🎉 انتهى اختبار النظام المحسن!")
    return True

if __name__ == "__main__":
    print("🚀 اختبار نظام معالجة الوثائق القانونية العربية المحسن")
    print("="*60)
    
    success = test_enhanced_backend()
    
    if success:
        print("\n✅ تم اختبار النظام بنجاح!")
    else:
        print("\n❌ فشل في اختبار النظام!") 