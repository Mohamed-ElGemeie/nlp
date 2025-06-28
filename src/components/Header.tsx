import React from 'react';
import { FileText, Brain, Search } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3 rtl:space-x-reverse">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                نظام البحث الذكي في المستندات العربية
              </h1>
              <p className="text-sm text-gray-600">
                Arabic PDF RAG System
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4 rtl:space-x-reverse">
            <div className="hidden md:flex items-center space-x-2 rtl:space-x-reverse text-sm text-gray-500">
              <FileText className="h-4 w-4" />
              <span>معالجة المستندات</span>
            </div>
            <div className="hidden md:flex items-center space-x-2 rtl:space-x-reverse text-sm text-gray-500">
              <Search className="h-4 w-4" />
              <span>البحث الدلالي</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;