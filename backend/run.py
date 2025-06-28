#!/usr/bin/env python3
"""
Startup script for the Arabic Legal Documents RAG API with Agentic Workflow
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_groq_api_key():
    """Check if Groq API key is set"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your-groq-api-key-here":
        print("⚠️ Groq API key not found!")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-actual-groq-api-key'")
        print("You can get a free API key from: https://console.groq.com/")
        return False
    else:
        print("✅ Groq API key found")
        return True

def check_tesseract():
    """Check if Tesseract is installed"""
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract is available")
            return True
        else:
            print("❌ Tesseract is not available")
            return False
    except FileNotFoundError:
        print("❌ Tesseract is not installed")
        print("Please install Tesseract OCR:")
        print("Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("macOS: brew install tesseract")
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def main():
    """Main startup function"""
    print("🚀 Starting Arabic Legal RAG System with Agentic Workflow")
    print("=" * 60)
    
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check Groq API key
    if not check_groq_api_key():
        print("\n⚠️ The system will work with limited functionality without Groq API key.")
    
    # Check Tesseract
    if not check_tesseract():
        print("\n⚠️ The system will work with limited OCR functionality without Tesseract.")
    
    print("\n🌟 Starting FastAPI server...")
    print("🔗 API will be available at: http://localhost:8001")
    print("📖 API docs at: http://localhost:8001/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start the FastAPI server
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()