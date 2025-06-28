import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import type { AgentWorkflowResponse } from './types';

// API configuration
const API_BASE_URL = 'http://localhost:8001';

// API functions
const chatAPI = async (message: string): Promise<AgentWorkflowResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

const uploadDocumentAPI = async (file: File): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

const checkHealthAPI = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
};

function App() {
  const [backendConnected, setBackendConnected] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    const initializeApp = async () => {
      const isHealthy = await checkHealthAPI();
      setBackendConnected(isHealthy);
    };

    initializeApp();
  }, []);

  const handleChat = async (message: string): Promise<AgentWorkflowResponse> => {
    if (!backendConnected) {
      throw new Error('Backend server is not available');
    }

    return await chatAPI(message);
  };

  const handleUpload = async (file: File): Promise<any> => {
    if (!backendConnected) {
      throw new Error('Backend server is not available');
    }

    return await uploadDocumentAPI(file);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3 rtl:space-x-reverse">
              <div className="bg-blue-600 p-2 rounded-lg">
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900" dir="rtl">
                  المساعد القانوني الذكي
                </h1>
                <p className="text-sm text-gray-600" dir="rtl">
                  نظام ذكي لتسجيل الشركات والامتثال القانوني
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4 rtl:space-x-reverse">
              <div className={`flex items-center ${backendConnected ? 'text-green-600' : 'text-red-600'}`}>
                <div className={`w-2 h-2 rounded-full mr-2 ${backendConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm">
                  {backendConnected ? 'متصل' : 'غير متصل'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Backend Connection Status */}
      {!backendConnected && (
        <div className="bg-yellow-50 border border-yellow-200 p-4">
          <div className="flex items-center text-yellow-700" dir="rtl">
            <div className="bg-yellow-100 p-2 rounded-full ml-3">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold">تحذير: الخادم الخلفي غير متصل</h3>
              <p className="text-yellow-600 mt-1">
                يرجى التأكد من تشغيل الخادم الخلفي على المنفذ 8001
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Interface */}
      <main className="container mx-auto px-6 py-8 max-w-4xl">
        <ChatInterface 
          onChat={handleChat}
          onUpload={handleUpload}
          backendConnected={backendConnected}
        />
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 text-white py-6 mt-16">
        <div className="container mx-auto px-6 text-center">
          <p className="text-gray-300" dir="rtl">
            المساعد القانوني الذكي - مدعوم بالذكاء الاصطناعي
          </p>
          <p className="text-gray-400 mt-2 text-sm" dir="rtl">
            نظام متقدم لتسجيل الشركات والامتثال القانوني في مصر
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;