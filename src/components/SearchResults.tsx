import React, { useState } from 'react';
import { FileText, Star, Copy, Check, ChevronDown, ChevronUp, Brain, Search, AlertTriangle, Users, User, Sparkles } from 'lucide-react';
import type { MultiAgentRAGResponse } from '../types';

interface SearchResultsProps {
  results: MultiAgentRAGResponse | null;
  searchType: 'search' | 'rag';
}

const SearchResults: React.FC<SearchResultsProps> = ({ 
  results, 
  searchType 
}) => {
  const [copiedStates, setCopiedStates] = useState<{ [key: string]: boolean }>({});
  const [expandedSources, setExpandedSources] = useState<{ [key: number]: boolean }>({});
  const [expandedAgents, setExpandedAgents] = useState<{ [key: number]: boolean }>({});

  if (!results) return null;

  if (results?.error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-6 mb-8">
        <div className="flex items-center text-red-700" dir="rtl">
          <div className="bg-red-100 p-2 rounded-full ml-3">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">خطأ في البحث</h3>
            <p className="text-red-600 mt-1">{results.error}</p>
          </div>
        </div>
      </div>
    );
  }

  const copyToClipboard = async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStates({ ...copiedStates, [key]: true });
      setTimeout(() => {
        setCopiedStates(prev => ({ ...prev, [key]: false }));
      }, 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const toggleSource = (index: number) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const toggleAgent = (index: number) => {
    setExpandedAgents(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1) + '%';
  };

  const formatProcessingTime = (time: number) => {
    return time < 1 ? `${(time * 1000).toFixed(0)}ms` : `${time.toFixed(1)}s`;
  };

  return (
    <div className="space-y-8">
      {/* Final AI Response Section */}
      {searchType === 'rag' && results?.final_answer && (
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl shadow-lg overflow-hidden">
          <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-6">
            <div className="flex items-center text-white" dir="rtl">
              <div className="bg-white/20 p-3 rounded-full ml-4">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-bold">الإجابة النهائية المدمجة</h3>
                <p className="text-green-100 mt-1">مُجمعة من نموذجين متقدمين للذكاء الاصطناعي</p>
              </div>
            </div>
          </div>
          
          <div className="p-6">
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <div className="flex items-start justify-between mb-4">
                <button
                  onClick={() => copyToClipboard(results?.final_answer || '', 'final-answer')}
                  className="flex items-center px-3 py-2 text-gray-500 hover:text-green-600 transition-colors duration-200"
                  title="نسخ الإجابة"
                >
                  {copiedStates['final-answer'] ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
                <h4 className="text-lg font-semibold text-gray-800" dir="rtl">
                  السؤال: {results?.question}
                </h4>
              </div>
              
              <div 
                className="prose prose-lg max-w-none text-right leading-relaxed text-gray-800 whitespace-pre-wrap min-h-[100px] max-h-[600px] overflow-y-auto"
                dir="rtl"
                lang="ar"
                style={{ 
                  fontFamily: '"Amiri", "Noto Naskh Arabic", "Times New Roman", serif',
                  unicodeBidi: 'embed',
                  direction: 'rtl',
                  textAlign: 'right'
                }}
              >
                {results?.final_answer}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Multi-Agent Responses Section */}
      {searchType === 'rag' && results?.agent_responses && results.agent_responses.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="bg-blue-50 p-6 border-b">
            <div className="flex items-center text-blue-700" dir="rtl">
              <div className="bg-blue-100 p-2 rounded-lg ml-3">
                <Users className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-xl font-bold">إجابات النماذج المتعددة</h3>
                <p className="text-blue-600 mt-1">
                  نموذجان متقدمان من الذكاء الاصطناعي قدما تحليلهما المتخصص
                </p>
              </div>
            </div>
          </div>

          <div className="divide-y divide-gray-100">
            {results.agent_responses.map((agent, index) => (
              <div key={index} className="p-6">
                <div className="flex items-start justify-between mb-4" dir="rtl">
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <div className="bg-purple-100 p-2 rounded-lg">
                      <User className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-800 text-lg">
                        {agent.model_name}
                      </h4>
                      <p className="text-gray-600 text-sm mt-1">
                        وقت المعالجة: {formatProcessingTime(agent.processing_time)}
                        {agent.error && (
                          <span className="text-red-500 mr-2">- خطأ: {agent.error}</span>
                        )}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <button
                      onClick={() => copyToClipboard(agent.response, `agent-${index}`)}
                      className="p-2 text-gray-400 hover:text-purple-600 transition-colors duration-200 rounded-lg hover:bg-purple-50"
                      title="نسخ الإجابة"
                    >
                      {copiedStates[`agent-${index}`] ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <button
                    onClick={() => toggleAgent(index)}
                    className="flex items-center justify-between w-full text-right p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200"
                    dir="rtl"
                  >
                    <div className="flex items-center">
                      {expandedAgents[index] ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </div>
                    <span className="font-medium text-gray-700">
                      {expandedAgents[index] ? 'إخفاء الإجابة' : 'عرض الإجابة'}
                    </span>
                  </button>
                </div>

                {expandedAgents[index] && (
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div 
                      className="text-gray-700 leading-relaxed whitespace-pre-wrap text-right"
                      dir="rtl"
                      lang="ar"
                      style={{ 
                        fontFamily: '"Amiri", "Noto Naskh Arabic", "Times New Roman", serif',
                        unicodeBidi: 'embed',
                        direction: 'rtl',
                        textAlign: 'right'
                      }}
                    >
                      {agent.response}
                    </div>
                  </div>
                )}

                {!expandedAgents[index] && (
                  <div 
                    className="text-gray-600 leading-relaxed text-right line-clamp-3"
                    dir="rtl"
                    lang="ar"
                    style={{ 
                      fontFamily: '"Amiri", "Noto Naskh Arabic", "Times New Roman", serif',
                      unicodeBidi: 'embed',
                      direction: 'rtl',
                      textAlign: 'right',
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden'
                    }}
                  >
                    {agent.response}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Orchestrator Response Section */}
      {searchType === 'rag' && results?.orchestrator_response && (
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-lg overflow-hidden">
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6">
            <div className="flex items-center text-white" dir="rtl">
              <div className="bg-white/20 p-3 rounded-full ml-4">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-bold">تحليل المُنسق (Orchestrator)</h3>
                <p className="text-indigo-100 mt-1">دمج وتحليل الإجابات المتعددة</p>
              </div>
            </div>
          </div>
          
          <div className="p-6">
            <div className="bg-white rounded-lg p-6 shadow-sm">
              <div className="flex items-start justify-between mb-4">
                <button
                  onClick={() => copyToClipboard(results?.orchestrator_response || '', 'orchestrator')}
                  className="flex items-center px-3 py-2 text-gray-500 hover:text-indigo-600 transition-colors duration-200"
                  title="نسخ تحليل المُنسق"
                >
                  {copiedStates['orchestrator'] ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
                <h4 className="text-lg font-semibold text-gray-800" dir="rtl">
                  تحليل المُنسق
                </h4>
              </div>
              
              <div 
                className="prose prose-lg max-w-none text-right leading-relaxed text-gray-800 whitespace-pre-wrap min-h-[100px] max-h-[400px] overflow-y-auto"
                dir="rtl"
                lang="ar"
                style={{ 
                  fontFamily: '"Amiri", "Noto Naskh Arabic", "Times New Roman", serif',
                  unicodeBidi: 'embed',
                  direction: 'rtl',
                  textAlign: 'right'
                }}
              >
                {results?.orchestrator_response}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search Results Section */}
      {results?.retrieved_documents && results.retrieved_documents.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="bg-gray-50 p-6 border-b">
            <div className="flex items-center justify-between" dir="rtl">
              <div className="flex items-center text-gray-700">
                <FileText className="w-5 h-5 ml-2" />
                <span className="font-medium">
                  تم العثور على {results.retrieved_documents.length} نتيجة مطابقة
                </span>
              </div>
              <span className="text-sm text-gray-500">
                مرتبة حسب درجة التطابق
              </span>
            </div>
          </div>

          <div className="divide-y divide-gray-100">
            {results.retrieved_documents.map((doc, index) => (
              <div key={index} className="p-6 hover:bg-gray-50 transition-colors duration-200">
                <div className="flex items-start justify-between mb-4" dir="rtl">
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <div className="bg-blue-100 p-2 rounded-lg">
                      <FileText className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-800 text-lg">
                        {doc.source}
                      </h4>
                      <p className="text-gray-600 text-sm mt-1">
                        {doc.metadata}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3 space-x-reverse">
                    <div className="flex items-center bg-green-100 px-3 py-1 rounded-full">
                      <Star className="w-4 h-4 text-green-600 ml-1" />
                      <span className="text-green-700 font-medium text-sm">
                        {formatScore(doc.similarity_score)}
                      </span>
                    </div>
                    
                    <button
                      onClick={() => copyToClipboard(doc.content, `doc-${index}`)}
                      className="p-2 text-gray-400 hover:text-blue-600 transition-colors duration-200 rounded-lg hover:bg-blue-50"
                      title="نسخ النص"
                    >
                      {copiedStates[`doc-${index}`] ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <button
                    onClick={() => toggleSource(index)}
                    className="flex items-center justify-between w-full text-right p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200"
                    dir="rtl"
                  >
                    <div className="flex items-center">
                      {expandedSources[index] ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </div>
                    <span className="font-medium text-gray-700">
                      {expandedSources[index] ? 'إخفاء المحتوى' : 'عرض المحتوى'}
                    </span>
                  </button>
                </div>

                {expandedSources[index] && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div 
                      className="text-gray-700 leading-relaxed whitespace-pre-wrap text-right"
                      dir="rtl"
                      lang="ar"
                      style={{ 
                        fontFamily: '"Amiri", "Noto Naskh Arabic", "Times New Roman", serif',
                        unicodeBidi: 'embed',
                        direction: 'rtl',
                        textAlign: 'right'
                      }}
                    >
                      {doc.content.length > 1000 ? (
                        <>
                          {doc.content.substring(0, 1000)}
                          <span className="text-gray-500">... (تم اقتطاع النص)</span>
                        </>
                      ) : (
                        doc.content
                      )}
                    </div>
                  </div>
                )}

                {!expandedSources[index] && (
                  <div 
                    className="text-gray-600 leading-relaxed text-right line-clamp-3"
                    dir="rtl"
                    lang="ar"
                    style={{ 
                      fontFamily: '"Amiri", "Noto Naskh Arabic", "Times New Roman", serif',
                      unicodeBidi: 'embed',
                      direction: 'rtl',
                      textAlign: 'right',
                      display: '-webkit-box',
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden'
                    }}
                  >
                    {doc.content}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchResults;