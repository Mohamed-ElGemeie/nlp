import React from 'react';
import { Brain, CheckCircle, Loader, Clock, Sparkles } from 'lucide-react';

interface AgentStatus {
  name: string;
  status: 'waiting' | 'running' | 'completed' | 'error';
  progress?: number;
  processingTime?: number;
  error?: string;
}

interface MultiAgentProgressProps {
  isVisible: boolean;
  currentAgent?: number;
  agents: AgentStatus[];
  orchestratorStatus?: 'waiting' | 'running' | 'completed' | 'error';
  totalProgress?: number;
}

const MultiAgentProgress: React.FC<MultiAgentProgressProps> = ({
  isVisible,
  currentAgent = 0,
  agents,
  orchestratorStatus = 'waiting',
  totalProgress = 0
}) => {
  if (!isVisible) return null;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Loader className="w-5 h-5 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
          <span className="text-white text-xs">!</span>
        </div>;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'border-blue-500 bg-blue-50';
      case 'completed':
        return 'border-green-500 bg-green-50';
      case 'error':
        return 'border-red-500 bg-red-50';
      default:
        return 'border-gray-300 bg-gray-50';
    }
  };

  const formatTime = (time?: number) => {
    if (!time) return '';
    return time < 1 ? `${(time * 1000).toFixed(0)}ms` : `${time.toFixed(1)}s`;
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border-2 border-blue-200">
      <div className="text-center mb-6">
        <div className="flex items-center justify-center mb-4">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-3 rounded-full">
            <Brain className="w-6 h-6 text-white" />
          </div>
        </div>
        <h3 className="text-2xl font-bold text-gray-800 mb-2" dir="rtl">
          التحليل متعدد النماذج
        </h3>
        <p className="text-gray-600" dir="rtl">
          يتم الآن تشغيل نموذجين متقدمين من الذكاء الاصطناعي لتحليل السؤال
        </p>
      </div>

      {/* Overall Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2" dir="rtl">
          <span className="text-sm font-medium text-gray-700">التقدم الإجمالي</span>
          <span className="text-sm text-gray-500">{totalProgress.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div 
            className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${totalProgress}%` }}
          ></div>
        </div>
      </div>

      {/* Agent Models Progress */}
      <div className="space-y-4 mb-6">
        <h4 className="text-lg font-semibold text-gray-800 text-right" dir="rtl">
          نماذج التحليل:
        </h4>
        
        {agents.map((agent, index) => (
          <div 
            key={index}
            className={`p-4 rounded-lg border-2 transition-all duration-300 ${getStatusColor(agent.status)}`}
          >
            <div className="flex items-center justify-between" dir="rtl">
              <div className="flex items-center space-x-3 space-x-reverse">
                {getStatusIcon(agent.status)}
                <div>
                  <h5 className="font-semibold text-gray-800">{agent.name}</h5>
                  <p className="text-sm text-gray-600">
                    {agent.status === 'waiting' && 'في الانتظار...'}
                    {agent.status === 'running' && 'جاري التحليل...'}
                    {agent.status === 'completed' && `تم الإنجاز في ${formatTime(agent.processingTime)}`}
                    {agent.status === 'error' && `خطأ: ${agent.error || 'غير معروف'}`}
                  </p>
                </div>
              </div>
              
              {agent.status === 'running' && (
                <div className="flex items-center space-x-2 space-x-reverse">
                  <div className="animate-pulse">
                    <div className="flex space-x-1 space-x-reverse">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              
              {agent.status === 'completed' && (
                <div className="text-green-600 text-sm font-medium">
                  ✅ مكتمل
                </div>
              )}
            </div>
            
            {agent.status === 'running' && agent.progress !== undefined && (
              <div className="mt-3">
                <div className="w-full bg-white bg-opacity-50 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${agent.progress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Orchestrator Status */}
      <div className={`p-4 rounded-lg border-2 transition-all duration-300 ${getStatusColor(orchestratorStatus)}`}>
        <div className="flex items-center justify-between" dir="rtl">
          <div className="flex items-center space-x-3 space-x-reverse">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h5 className="font-semibold text-gray-800">منسق الإجابات (Orchestrator)</h5>
              <p className="text-sm text-gray-600">
                {orchestratorStatus === 'waiting' && 'في انتظار اكتمال النماذج...'}
                {orchestratorStatus === 'running' && 'جاري دمج وتحليل الإجابات...'}
                {orchestratorStatus === 'completed' && 'تم دمج الإجابات بنجاح'}
                {orchestratorStatus === 'error' && 'خطأ في دمج الإجابات'}
              </p>
            </div>
          </div>
          
          {orchestratorStatus === 'running' && (
            <div className="flex items-center">
              <div className="animate-spin">
                <Sparkles className="w-6 h-6 text-indigo-600" />
              </div>
            </div>
          )}
          
          {orchestratorStatus === 'completed' && (
            <div className="text-green-600 text-sm font-medium">
              🎯 مكتمل
            </div>
          )}
        </div>
      </div>

      {/* Estimated Time */}
      <div className="mt-4 text-center">
        <p className="text-sm text-gray-500" dir="rtl">
          ⏱️ الوقت المتوقع: 1-2 دقيقة للحصول على أفضل النتائج
        </p>
      </div>
    </div>
  );
};

export default MultiAgentProgress; 