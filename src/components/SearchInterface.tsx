import React, { useState } from 'react';
import { Search, MessageSquare, Sparkles, ArrowLeft } from 'lucide-react';
import type { MultiAgentRAGResponse } from '../types';

interface SearchInterfaceProps {
  onSearch: (query: string) => Promise<MultiAgentRAGResponse>;
  isSearching: boolean;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ onSearch, isSearching }) => {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<'search' | 'rag'>('rag');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isSearching) {
      await onSearch(query.trim());
    }
  };

  const suggestedQuestions = [
    'ما هي الضرائب المفروضة على الشركات؟',
    'ما هي إجراءات تأسيس شركة جديدة؟',
    'ما هي حقوق العامل في قانون العمل؟',
    'ما هي أنواع العقود المدنية؟',
    'ما هي شروط الحصول على براءة اختراع؟',
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-4" dir="rtl">
          اسأل عن أي معلومة قانونية
        </h2>
        <p className="text-gray-600 text-lg" dir="rtl">
          استخدم الذكاء الاصطناعي للحصول على إجابات دقيقة من الوثائق القانونية
        </p>
      </div>

      <div className="flex justify-center mb-6">
        <div className="bg-gray-100 rounded-lg p-1 flex">
          <button
            type="button"
            onClick={() => setSearchType('rag')}
            className={`px-6 py-2 rounded-md font-medium transition-all duration-200 flex items-center ${
              searchType === 'rag'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <Sparkles className="w-4 h-4 ml-2" />
            إجابة ذكية
          </button>
          <button
            type="button"
            onClick={() => setSearchType('search')}
            className={`px-6 py-2 rounded-md font-medium transition-all duration-200 flex items-center ${
              searchType === 'search'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <Search className="w-4 h-4 ml-2" />
            بحث عادي
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              searchType === 'rag'
                ? 'اكتب سؤالك هنا وسيقوم النظام بالبحث في الوثائق وتقديم إجابة شاملة...'
                : 'ابحث في محتوى الوثائق...'
            }
            className="w-full h-32 p-4 text-right border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-100 outline-none transition-all duration-200 resize-none text-lg"
            dir="rtl"
            disabled={isSearching}
          />
          <div className="absolute bottom-4 left-4">
            {searchType === 'rag' ? (
              <MessageSquare className="w-6 h-6 text-gray-400" />
            ) : (
              <Search className="w-6 h-6 text-gray-400" />
            )}
          </div>
        </div>

        <div className="flex justify-center">
          <button
            type="submit"
            disabled={!query.trim() || isSearching}
            className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-8 py-4 rounded-xl font-medium text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:from-blue-700 hover:to-blue-800 transition-all duration-200 flex items-center shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            {isSearching ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white ml-3"></div>
                جاري البحث...
              </>
            ) : (
              <>
                {searchType === 'rag' ? (
                  <Sparkles className="w-5 h-5 ml-3" />
                ) : (
                  <Search className="w-5 h-5 ml-3" />
                )}
                {searchType === 'rag' ? 'احصل على إجابة ذكية' : 'ابحث الآن'}
              </>
            )}
          </button>
        </div>
      </form>

      <div className="mt-8">
        <h3 className="text-lg font-semibold text-gray-700 mb-4 text-right" dir="rtl">
          أسئلة مقترحة:
        </h3>
        <div className="grid gap-3">
          {suggestedQuestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => setQuery(suggestion)}
              className="text-right p-4 bg-gray-50 hover:bg-blue-50 rounded-lg transition-colors duration-200 text-gray-700 hover:text-blue-700 border border-transparent hover:border-blue-200"
              dir="rtl"
              disabled={isSearching}
            >
              <div className="flex items-center justify-between">
                <ArrowLeft className="w-4 h-4 text-gray-400" />
                <span>{suggestion}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SearchInterface;