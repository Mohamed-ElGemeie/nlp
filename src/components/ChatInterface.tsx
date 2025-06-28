import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, FileText, Loader, CheckCircle, AlertTriangle, User, Bot, Search, FileCheck } from 'lucide-react';
import type { AgentWorkflowResponse } from '../types';

interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
  agentResponse?: AgentWorkflowResponse;
}

interface ChatInterfaceProps {
  onChat: (message: string) => Promise<AgentWorkflowResponse>;
  onUpload: (file: File) => Promise<any>;
  backendConnected: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onChat, onUpload, backendConnected }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add welcome message on first load
  useEffect(() => {
    if (messages.length === 0) {
      const welcomeMessage: ChatMessage = {
        id: 'welcome',
        type: 'agent',
        content: `مرحباً بك في المساعد القانوني الذكي! 🏛️

أنا هنا لمساعدتك في تسجيل شركتك والامتثال للقوانين المصرية. 

سأقوم بجمع المعلومات اللازمة عن شركتك، والبحث في قاعدة البيانات القانونية، وتقديم التوصيات المناسبة.

يمكنك البدء بإخباري عن نوع الشركة التي تريد تسجيلها أو أي استفسار قانوني لديك.`,
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, []);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !backendConnected) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await onChat(inputMessage);
      
      const agentMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: response.final_recommendation || 'تم معالجة طلبك بنجاح.',
        timestamp: new Date(),
        agentResponse: response
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: `عذراً، حدث خطأ أثناء معالجة طلبك: ${error instanceof Error ? error.message : 'خطأ غير معروف'}`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !backendConnected) return;

    if (!file.name.endsWith('.pdf')) {
      setUploadStatus('❌ يرجى رفع ملفات PDF فقط');
      return;
    }

    setUploadStatus('📤 جاري رفع الملف...');

    try {
      const result = await onUpload(file);
      setUploadStatus(`✅ ${result.message || 'تم رفع الملف بنجاح'}`);
      
      // Add upload confirmation message
      const uploadMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'agent',
        content: `تم رفع الوثيقة "${file.name}" بنجاح وإضافتها إلى قاعدة البيانات. ${result.chunks_created ? `تم إنشاء ${result.chunks_created} قسم نصي.` : ''}`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, uploadMessage]);
      
      setTimeout(() => setUploadStatus(''), 3000);
    } catch (error) {
      setUploadStatus(`❌ فشل في رفع الملف: ${error instanceof Error ? error.message : 'خطأ غير معروف'}`);
      setTimeout(() => setUploadStatus(''), 5000);
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('ar-EG', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const renderAgentResponse = (response: AgentWorkflowResponse) => {
    return (
      <div className="mt-4 space-y-4">
        {/* Collected Information */}
        {response.collected_info && Object.keys(response.collected_info).length > 0 && (
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 mb-2 flex items-center" dir="rtl">
              <User className="w-4 h-4 ml-2" />
              المعلومات المجمعة
            </h4>
            <div className="text-sm text-blue-700 space-y-1" dir="rtl">
              {Object.entries(response.collected_info).map(([key, value]) => (
                <div key={key}>
                  <span className="font-medium">{key}:</span> {value}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Search Items */}
        {response.search_items && response.search_items.length > 0 && (
          <div className="bg-purple-50 rounded-lg p-4">
            <h4 className="font-semibold text-purple-800 mb-2 flex items-center" dir="rtl">
              <Search className="w-4 h-4 ml-2" />
              نقاط البحث المحددة
            </h4>
            <div className="space-y-2">
              {response.search_items.map((item, index) => (
                <div key={index} className="bg-white rounded p-2 text-sm">
                  <div className="flex items-center justify-between" dir="rtl">
                    <span className="text-purple-700">{item.query}</span>
                    <div className="flex items-center space-x-2 space-x-reverse">
                      <span className="text-xs bg-purple-100 px-2 py-1 rounded">{item.category}</span>
                      <span className="text-xs text-purple-600">أولوية: {item.priority}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* RAG Results */}
        {response.rag_results && response.rag_results.length > 0 && (
          <div className="bg-green-50 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 mb-2 flex items-center" dir="rtl">
              <FileCheck className="w-4 h-4 ml-2" />
              نتائج البحث القانوني
            </h4>
            <div className="space-y-3">
              {response.rag_results.map((result, index) => (
                <div key={index} className="bg-white rounded p-3">
                  <h5 className="font-medium text-green-800 mb-2" dir="rtl">
                    {result.search_item}
                  </h5>
                  <p className="text-sm text-green-700 mb-2" dir="rtl">
                    {result.summary}
                  </p>
                  {result.documents && result.documents.length > 0 && (
                    <div className="text-xs text-green-600" dir="rtl">
                      تم العثور على {result.documents.length} وثيقة ذات صلة
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processing Time */}
        {response.processing_time && (
          <div className="text-xs text-gray-500 text-center">
            وقت المعالجة: {response.processing_time.toFixed(2)} ثانية
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-xl shadow-lg h-[600px] flex flex-col">
      {/* Chat Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-t-xl">
        <div className="flex items-center justify-between" dir="rtl">
          <div className="flex items-center space-x-3 space-x-reverse">
            <div className="bg-white/20 p-2 rounded-full">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold">المساعد القانوني</h3>
              <p className="text-blue-100 text-sm">متخصص في تسجيل الشركات والامتثال القانوني</p>
            </div>
          </div>
          
          {/* Upload Button */}
          <div className="flex items-center space-x-2 space-x-reverse">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!backendConnected}
              className="bg-white/20 hover:bg-white/30 p-2 rounded-lg transition-colors duration-200 disabled:opacity-50"
              title="رفع وثيقة PDF"
            >
              <Upload className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        {/* Upload Status */}
        {uploadStatus && (
          <div className="mt-2 text-sm bg-white/10 rounded p-2" dir="rtl">
            {uploadStatus}
          </div>
        )}
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-start' : 'justify-end'}`}
            dir="rtl"
          >
            <div
              className={`max-w-[80%] rounded-lg p-4 ${
                message.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="flex items-start space-x-3 space-x-reverse">
                <div className={`p-2 rounded-full ${
                  message.type === 'user' ? 'bg-blue-500' : 'bg-gray-200'
                }`}>
                  {message.type === 'user' ? (
                    <User className="w-4 h-4" />
                  ) : (
                    <Bot className="w-4 h-4" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="whitespace-pre-wrap text-right" dir="rtl">
                    {message.content}
                  </div>
                  
                  {/* Render agent response details */}
                  {message.type === 'agent' && message.agentResponse && (
                    renderAgentResponse(message.agentResponse)
                  )}
                  
                  <div className={`text-xs mt-2 ${
                    message.type === 'user' ? 'text-blue-200' : 'text-gray-500'
                  }`}>
                    {formatTime(message.timestamp)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-end" dir="rtl">
            <div className="bg-gray-100 rounded-lg p-4 max-w-[80%]">
              <div className="flex items-center space-x-3 space-x-reverse">
                <div className="bg-gray-200 p-2 rounded-full">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="flex items-center space-x-2 space-x-reverse">
                  <Loader className="w-4 h-4 animate-spin text-blue-600" />
                  <span className="text-gray-600">جاري التحليل والبحث...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t p-4">
        <div className="flex items-center space-x-3 space-x-reverse">
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading || !backendConnected}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white p-3 rounded-lg transition-colors duration-200"
          >
            {isLoading ? (
              <Loader className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
          
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={backendConnected ? "اكتب رسالتك هنا..." : "الخادم غير متصل"}
            disabled={isLoading || !backendConnected}
            className="flex-1 border border-gray-300 rounded-lg p-3 text-right resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
            dir="rtl"
            rows={2}
          />
        </div>
        
        <div className="mt-2 text-xs text-gray-500 text-center" dir="rtl">
          اضغط Enter للإرسال • Shift+Enter لسطر جديد
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;