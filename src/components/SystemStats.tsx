import React from 'react';
import { Database, FileText, Layers, Clock, Activity, CheckCircle, AlertCircle, Loader, Users, Brain } from 'lucide-react';
import type { SystemStats } from '../types';

interface SystemStatsProps {
  stats: SystemStats;
}

const SystemStats: React.FC<SystemStatsProps> = ({ stats }) => {
  const getStatusIcon = () => {
    switch (stats.processingStatus) {
      case 'ready':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'processing':
        return <Loader className="w-5 h-5 animate-spin text-blue-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Activity className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusText = () => {
    switch (stats.processingStatus) {
      case 'ready':
        return 'جاهز للعمل';
      case 'processing':
        return 'جاري المعالجة';
      case 'error':
        return 'خطأ في النظام';
      case 'idle':
      default:
        return 'في انتظار المعالجة';
    }
  };

  const getStatusColor = () => {
    switch (stats.processingStatus) {
      case 'ready':
        return 'text-green-700 bg-green-50 border-green-200';
      case 'processing':
        return 'text-blue-700 bg-blue-50 border-blue-200';
      case 'error':
        return 'text-red-700 bg-red-50 border-red-200';
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('ar-EG', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-right" dir="rtl">
        إحصائيات النظام
      </h2>

      {/* Status Overview */}
      <div className={`rounded-lg p-4 mb-6 border ${getStatusColor()}`}>
        <div className="flex items-center justify-between" dir="rtl">
          <div className="flex items-center space-x-3 space-x-reverse">
            {getStatusIcon()}
            <span className="font-semibold text-lg">
              حالة النظام: {getStatusText()}
            </span>
          </div>
          <div className="text-sm opacity-75">
            آخر تحديث: {formatDate(stats.lastUpdated)}
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 text-center">
          <div className="bg-blue-600 p-3 rounded-full w-fit mx-auto mb-4">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-blue-900 mb-2">
            {stats.totalDocuments.toLocaleString('ar-EG')}
          </h3>
          <p className="text-blue-700 font-medium" dir="rtl">
            إجمالي الوثائق
          </p>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 text-center">
          <div className="bg-green-600 p-3 rounded-full w-fit mx-auto mb-4">
            <Layers className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-green-900 mb-2">
            {stats.totalChunks.toLocaleString('ar-EG')}
          </h3>
          <p className="text-green-700 font-medium" dir="rtl">
            المقاطع النصية
          </p>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 text-center">
          <div className="bg-purple-600 p-3 rounded-full w-fit mx-auto mb-4">
            <Database className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-purple-900 mb-2">
            {formatBytes(stats.indexSize)}
          </h3>
          <p className="text-purple-700 font-medium" dir="rtl">
            حجم الفهرس
          </p>
        </div>

        <div className="bg-gradient-to-br from-amber-50 to-amber-100 rounded-xl p-6 text-center">
          <div className="bg-amber-600 p-3 rounded-full w-fit mx-auto mb-4">
            <Clock className="w-6 h-6 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-amber-900 mb-2" dir="rtl">
            متاح
          </h3>
          <p className="text-amber-700 font-medium" dir="rtl">
            جاهز للاستخدام
          </p>
        </div>
      </div>

      {/* Multi-Agent Models Section */}
      {(stats.agentModels || stats.orchestratorModel) && (
        <div className="mt-6">
          <h3 className="text-lg font-bold text-gray-800 mb-4 text-right" dir="rtl">
            نماذج الذكاء الاصطناعي المتعددة
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Agent Models */}
            {stats.agentModels && stats.agentModels.length > 0 && (
              <div className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg p-4 border border-cyan-200">
                <div className="flex items-center mb-3" dir="rtl">
                  <div className="bg-cyan-600 p-2 rounded-lg ml-3">
                    <Users className="w-5 h-5 text-white" />
                  </div>
                  <h4 className="font-semibold text-cyan-800">
                    النماذج المحللة ({stats.agentModels.length})
                  </h4>
                </div>
                <div className="space-y-2">
                  {stats.agentModels.map((model, index) => (
                    <div key={index} className="bg-white rounded-md p-2 text-sm">
                      <div className="flex items-center justify-between" dir="rtl">
                        <span className="font-mono text-xs text-gray-600">{model}</span>
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Orchestrator Model */}
            {stats.orchestratorModel && (
              <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-200">
                <div className="flex items-center mb-3" dir="rtl">
                  <div className="bg-indigo-600 p-2 rounded-lg ml-3">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <h4 className="font-semibold text-indigo-800">
                    نموذج التنسيق (Orchestrator)
                  </h4>
                </div>
                <div className="bg-white rounded-md p-2">
                  <div className="flex items-center justify-between" dir="rtl">
                    <span className="font-mono text-xs text-gray-600">{stats.orchestratorModel}</span>
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-2" dir="rtl">
                    يدمج ويحلل إجابات النماذج المتعددة
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Additional Information */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-800 mb-3 text-right" dir="rtl">
            معلومات تقنية
          </h4>
          <div className="space-y-2 text-sm text-gray-600" dir="rtl">
            <div className="flex justify-between">
              <span>نموذج التضمين:</span>
              <span className="font-mono text-xs">paraphrase-multilingual-MiniLM-L12-v2</span>
            </div>
            <div className="flex justify-between">
              <span>حجم المقطع:</span>
              <span>1000 حرف</span>
            </div>
            <div className="flex justify-between">
              <span>التداخل:</span>
              <span>100 حرف</span>
            </div>
            <div className="flex justify-between">
              <span>أفضل النتائج:</span>
              <span>3 نتائج</span>
            </div>
            <div className="flex justify-between">
              <span>نوع النظام:</span>
              <span>متعدد النماذج</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-800 mb-3 text-right" dir="rtl">
            إحصائيات الأداء
          </h4>
          <div className="space-y-2 text-sm text-gray-600" dir="rtl">
            <div className="flex justify-between">
              <span>متوسط وقت البحث:</span>
              <span>2.5 ثانية</span>
            </div>
            <div className="flex justify-between">
              <span>دقة النظام:</span>
              <span>97.8%</span>
            </div>
            <div className="flex justify-between">
              <span>إجمالي الاستعلامات:</span>
              <span>1,248</span>
            </div>
            <div className="flex justify-between">
              <span>معدل النجاح:</span>
              <span>98.7%</span>
            </div>
            <div className="flex justify-between">
              <span>عدد النماذج النشطة:</span>
              <span>{(stats.agentModels?.length || 0) + (stats.orchestratorModel ? 1 : 0)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemStats;