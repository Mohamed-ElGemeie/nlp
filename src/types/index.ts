export interface SearchItem {
  query: string;
  priority: number;
  category: string;
}

export interface RAGResult {
  search_item: string;
  documents: Array<{
    content: string;
    source: string;
    metadata: string;
    similarity_score: number;
    chunk_id: number;
  }>;
  summary: string;
}

export interface AgentResponse {
  agent_name: string;
  response: string;
  processing_time: number;
  error?: string;
}

export interface AgentWorkflowResponse {
  collected_info: Record<string, string>;
  search_items: SearchItem[];
  rag_results: RAGResult[];
  final_recommendation: string;
  processing_time: number;
  error?: string;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
  agentResponse?: AgentWorkflowResponse;
}