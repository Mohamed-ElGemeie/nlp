import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import SearchInterface from './components/SearchInterface';
import SearchResults from './components/SearchResults';
import SystemStats from './components/SystemStats';
import MultiAgentProgress from './components/MultiAgentProgress';
import type { RAGResponse, MultiAgentRAGResponse, UploadedFile, SystemStats as SystemStatsType } from './types';

// API configuration
const API_BASE_URL = 'http://localhost:8001';

// Multi-Agent Models Configuration
const AGENT_MODELS = [
  { name: 'llama3:latest', displayName: 'LLaMA 3' },
  { name: 'qwen2.5:14b', displayName: 'Qwen 2.5 (14B)' }
];

interface AgentStatus {
  name: string;
  status: 'waiting' | 'running' | 'completed' | 'error';
  progress?: number;
  processingTime?: number;
  error?: string;
}

// API functions
const searchAPI = async (query: string): Promise<MultiAgentRAGResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, top_k: 5 }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// Simple search API that returns final results (no streaming)
const searchStreamAPI = async (query: string): Promise<MultiAgentRAGResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/search/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, top_k: 5 }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

const getSystemStatsAPI = async (): Promise<SystemStatsType> => {
  const response = await fetch(`${API_BASE_URL}/api/stats`);
  
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
  const [searchResults, setSearchResults] = useState<MultiAgentRAGResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [searchType] = useState<'search' | 'rag'>('rag');
  const [systemStats, setSystemStats] = useState<SystemStatsType | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([]);
  const [orchestratorStatus, setOrchestratorStatus] = useState<'waiting' | 'running' | 'completed' | 'error'>('waiting');
  const [totalProgress, setTotalProgress] = useState(0);

  // Check backend health and load system stats on mount
  useEffect(() => {
    const initializeApp = async () => {
      // Check if backend is available
      const isHealthy = await checkHealthAPI();
      setBackendConnected(isHealthy);

      if (isHealthy) {
        try {
          const stats = await getSystemStatsAPI();
          setSystemStats(stats);
        } catch (error) {
          console.error('Failed to load system stats:', error);
        }
      }
    };

    initializeApp();
  }, []);

  // Initialize agent statuses
  const initializeAgentStatuses = () => {
    const initialStatuses: AgentStatus[] = AGENT_MODELS.map(model => ({
      name: model.displayName,
      status: 'waiting' as const
    }));
    setAgentStatuses(initialStatuses);
    setOrchestratorStatus('waiting');
    setTotalProgress(0);
  };

  // Simulate progress updates for visual feedback
  const simulateAgentProgress = async (agentIndex: number, delay: number = 2000) => {
    // Set agent to running
    setAgentStatuses(prev => prev.map((agent, index) => 
      index === agentIndex 
        ? { ...agent, status: 'running' as const, progress: 0 }
        : agent
    ));

    // Simulate progress
    const progressSteps = 10;
    const stepDelay = delay / progressSteps;
    
    for (let step = 1; step <= progressSteps; step++) {
      await new Promise(resolve => setTimeout(resolve, stepDelay));
      const progress = (step / progressSteps) * 100;
      
      setAgentStatuses(prev => prev.map((agent, index) => 
        index === agentIndex 
          ? { ...agent, progress }
          : agent
      ));
      
      // Update total progress
      const totalSteps = AGENT_MODELS.length + 1; // +1 for orchestrator
      const currentProgress = ((agentIndex + (step / progressSteps)) / totalSteps) * 100;
      setTotalProgress(currentProgress);
    }

    // Mark as completed
    setAgentStatuses(prev => prev.map((agent, index) => 
      index === agentIndex 
        ? { ...agent, status: 'completed' as const, processingTime: delay / 1000 }
        : agent
    ));
  };

  const handleSearch = async (query: string): Promise<MultiAgentRAGResponse> => {
    setIsSearching(true);
    setSearchResults(null);
    
    // Initialize progress tracking
    initializeAgentStatuses();
    
    try {
      if (!backendConnected) {
        throw new Error('Backend server is not available');
      }

      // Start visual progress simulation
      const startTime = Date.now();
      
      // Simulate agent progress (this runs parallel to actual API call)
      const progressPromises = AGENT_MODELS.map((_, index) => 
        simulateAgentProgress(index, 2000 + (index * 800)) // Stagger the timing for 2 models
      );

      // Start all progress simulations
      Promise.all(progressPromises).then(() => {
        // Start orchestrator after all agents
        setOrchestratorStatus('running');
        setTimeout(() => {
          setOrchestratorStatus('completed');
          setTotalProgress(100);
        }, 1500);
      });

      // Make the actual API call
      const results = searchType === 'rag' ? await searchStreamAPI(query) : await searchAPI(query);
      
      // Ensure minimum time for visual feedback
      const elapsedTime = Date.now() - startTime;
      const minTime = 4000; // Minimum 4 seconds for visual effect with 2 models
      if (elapsedTime < minTime) {
        await new Promise(resolve => setTimeout(resolve, minTime - elapsedTime));
      }

      setSearchResults(results);
      
      // Refresh system stats after search
      try {
        const stats = await getSystemStatsAPI();
        setSystemStats(stats);
      } catch (error) {
        console.error('Failed to refresh system stats:', error);
      }
      
      return results;
    } catch (error) {
      console.error('Search error:', error);
      
      // Mark current agent as error
      setAgentStatuses(prev => prev.map(agent => 
        agent.status === 'running' 
          ? { ...agent, status: 'error' as const, error: 'فشل في التحليل' }
          : agent
      ));
      setOrchestratorStatus('error');
      
      const errorResponse: MultiAgentRAGResponse = {
        question: query,
        retrieved_documents: [],
        agent_responses: [],
        orchestrator_response: '',
        final_answer: '',
        error: backendConnected 
          ? 'حدث خطأ أثناء البحث. يرجى المحاولة مرة أخرى.' 
          : 'الخادم غير متوفر. يرجى التأكد من تشغيل الخادم الخلفي على المنفذ 8001.'
      };
      setSearchResults(errorResponse);
      return errorResponse;
    } finally {
      setIsSearching(false);
    }
  };

  const handleFilesUploaded = (files: UploadedFile[]) => {
    setUploadedFiles(files);
  };

  // Default system stats for when backend is not connected
  const defaultStats: SystemStatsType = {
    totalDocuments: 0,
    totalChunks: 0,
    indexSize: 0,
    lastUpdated: new Date().toISOString(),
    processingStatus: backendConnected ? 'ready' : 'error'
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <Header />
      
      <main className="container mx-auto px-6 py-8 max-w-7xl">
        {/* Backend Connection Status */}
        {!backendConnected && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 mb-8">
            <div className="flex items-center text-yellow-700" dir="rtl">
              <div className="bg-yellow-100 p-2 rounded-full ml-3">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-lg">تحذير: الخادم الخلفي غير متصل</h3>
                <p className="text-yellow-600 mt-1">
                  يرجى التأكد من تشغيل الخادم الخلفي على المنفذ 8001. قم بتشغيل: python backend/main.py
                </p>
              </div>
            </div>
          </div>
        )}

        {/* System Statistics */}
        <SystemStats stats={systemStats || defaultStats} />
        
        {/* File Upload Section - Only show if backend is connected */}
        {backendConnected && (
          <FileUpload 
            onFilesUploaded={handleFilesUploaded}
            uploadedFiles={uploadedFiles}
          />
        )}
        
        {/* Search Interface */}
        <SearchInterface 
          onSearch={handleSearch}
          isSearching={isSearching}
        />
        
        {/* Multi-Agent Progress */}
        <MultiAgentProgress
          isVisible={isSearching}
          agents={agentStatuses}
          orchestratorStatus={orchestratorStatus}
          totalProgress={totalProgress}
        />
        
        {/* Search Results */}
        {searchResults && (
          <SearchResults 
            results={searchResults}
            searchType={searchType}
          />
        )}
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="container mx-auto px-6 text-center">
          <p className="text-gray-300" dir="rtl">
            نظام البحث الذكي في الوثائق العربية - مدعوم بالذكاء الاصطناعي
          </p>
          <p className="text-gray-400 mt-2 text-sm">
            تم تطويره لمعالجة وفهرسة النصوص القانونية والوثائق الرسمية باللغة العربية
          </p>
          <div className="mt-4 flex items-center justify-center space-x-4 space-x-reverse">
            <div className={`flex items-center ${backendConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${backendConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">
                {backendConnected ? 'متصل بالخادم' : 'غير متصل بالخادم'}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;