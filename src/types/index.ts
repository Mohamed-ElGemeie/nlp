export interface Document {
  id: string;
  content: string;
  source: string;
  chunk_id: number;
  metadata: string;
  embedding?: number[];
}

export interface SearchResult {
  content: string;
  source: string;
  metadata: string;
  similarity_score: number;
  chunk_id: number;
}

export interface AgentResponse {
  model_name: string;
  response: string;
  processing_time: number;
  error?: string;
}

export interface RAGResponse {
  question: string;
  retrieved_documents: SearchResult[];
  model_answer: string;
  processing_time?: number;
  error?: string;
}

export interface MultiAgentRAGResponse {
  question: string;
  retrieved_documents: SearchResult[];
  agent_responses: AgentResponse[];
  orchestrator_response: string;
  final_answer: string;
  total_processing_time?: number;
  error?: string;
}

export interface ProcessingStatus {
  filename: string;
  status: 'processing' | 'completed' | 'error';
  chunks: number;
  progress: number;
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  uploadedAt: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  chunks?: number;
}

export interface SystemStats {
  totalDocuments: number;
  totalChunks: number;
  indexSize: number;
  lastUpdated: string;
  processingStatus: 'idle' | 'processing' | 'ready' | 'error';
  agentModels?: string[];
  orchestratorModel?: string;
}