#!/usr/bin/env python3
"""
Startup script for the Arabic Legal Documents RAG API
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

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is available")
            return True
        else:
            print("❌ Ollama is not available")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed or not in PATH")
        return False

def check_deepseek_model():
    """Check if deepseek-r1:7b model is available in Ollama"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "deepseek-r1:7b" in result.stdout:
            print("✅ deepseek-r1:7b model is available")
            return True
        else:
            print("⚠️ deepseek-r1:7b model not found")
            print("📥 Downloading deepseek-r1:7b model...")
            subprocess.run(["ollama", "pull", "deepseek-r1:7b"])
            return True
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return False

def main():
    """Main startup function"""
    print("🚀 Starting Arabic Legal Documents RAG API")
    print("=" * 50)
    
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n📋 To install Ollama:")
        print("1. Visit: https://ollama.ai/")
        print("2. Download and install Ollama")
        print("3. Run: ollama pull deepseek-r1:7b")
        print("\n⚠️ The system will work without Ollama, but AI responses will show error messages.")
    else:
        check_deepseek_model()
    
    print("\n🌟 Starting FastAPI server...")
    print("🔗 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the FastAPI server
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 