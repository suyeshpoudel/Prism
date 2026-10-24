export interface PageExtraction {
  page_number: number;
  text: string;
  char_count: number;
  is_empty: boolean;
}

export interface Chunk {
  id: string;
  document_id: string;
  page_number: number;
  chunk_index: number;
  text: string;
  char_count: number;
  start_offset: number;
  end_offset: number;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: string;
}

export interface DocumentDetail extends DocumentSummary {
  content_type: string | null;
  extracted_char_count: number;
  pages: PageExtraction[];
  chunks: Chunk[];
  warnings: string[];
}

export interface SearchResult {
  score: number;
  chunk: Chunk;
  document_id: string;
  filename: string;
}

export interface SearchResponse {
  query: string;
  top_k: number;
  collection_id: string;
  results: SearchResult[];
}

export interface Citation {
  marker: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  text: string;
  score: number;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  insufficient_evidence: boolean;
  citations: Citation[];
  invalid_citation_markers: number[];
  evidence: SearchResult[];
  top_k: number;
  collection_id: string;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
}

export interface BenchmarkDocument {
  filename: string;
  page_count: number;
  char_count: number;
}

export interface BenchmarkCase {
  id: string;
  question: string;
  relevant: { filename: string; page_number: number }[];
  k_values: number[];
}

export interface BenchmarkInfo {
  documents: BenchmarkDocument[];
  cases: BenchmarkCase[];
}

export interface RetrievalConfiguration {
  chunk_size: number;
  chunk_overlap: number;
  embedding_model?: string | null;
}

export type MetricMap = Record<string, number | null>;

export interface RetrievalReport {
  type: "retrieval";
  config: {
    embedding_model: string;
    chunk_size: number;
    chunk_overlap: number;
    corpus_documents: number;
    corpus_chunks: number;
    cases: number;
    k_values: number[];
  };
  aggregate: MetricMap;
  cases: {
    id: string;
    question: string;
    metrics: MetricMap;
    timings_ms: { embedding: number; retrieval: number; total: number };
    top_results: {
      rank: number;
      filename: string;
      page_number: number;
      chunk_id: string;
      score: number;
    }[];
  }[];
}

export interface EvaluationRecord {
  id: string;
  created_at: string;
  owner_id?: string;
  type: "retrieval" | "answer";
  config: Record<string, unknown>;
  comparison?: MetricMap[];
  results?: RetrievalReport[];
  aggregate?: MetricMap;
  method_notes?: Record<string, string>;
  usage?: {
    input_tokens: number | null;
    output_tokens: number | null;
    estimated_cost_usd: number | null;
    input_cost_per_1m?: number;
    output_cost_per_1m?: number;
  };
  cases?: AnswerCaseReport[];
  timings_ms?: Record<string, number>;
}

export interface AnswerCaseReport {
  id: string;
  question: string;
  relevant: { filename: string; page_number: number }[];
  answer?: string;
  insufficient_evidence?: boolean;
  model?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  metrics: MetricMap;
  citation_details?: {
    valid_markers: number[];
    invalid_markers: number[];
    cited_keys: { filename: string; page_number: number }[];
    sentences: number;
    cited_sentences: number;
    unsupported_cited_sentences: number;
  };
  top_results?: {
    rank: number;
    filename: string;
    page_number: number;
    chunk_id: string;
    score: number;
  }[];
  llm_judge?: {
    model?: string;
    faithful?: boolean;
    unsupported_claims?: string[];
    reason?: string;
    error?: string;
    timings_ms?: number;
  };
  error?: string;
  timings_ms?: { embedding?: number; retrieval?: number; generation?: number; total?: number };
}

export interface RunSummary {
  id: string;
  created_at: string | null;
  type: "retrieval" | "answer" | null;
  config: Record<string, unknown> | null;
  aggregate: MetricMap | null;
  configurations: number | null;
  comparison: MetricMap[] | null;
}
