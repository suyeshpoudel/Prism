import type {
  AnswerResponse,
  BenchmarkInfo,
  DocumentDetail,
  DocumentSummary,
  EvaluationRecord,
  RetrievalConfiguration,
  RunSummary,
  SearchResponse,
} from "./types";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_PRISM_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

interface RequestOptions {
  apiKey: string;
  method?: "GET" | "POST" | "DELETE";
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions): Promise<T> {
  const headers: Record<string, string> = { "X-API-Key": options.apiKey };
  let body: BodyInit | undefined;
  if (options.form) {
    body = options.form;
  } else if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method ?? "GET",
      headers,
      body,
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      0,
      "network_error",
      `Could not reach the Prism API at ${API_BASE_URL}. Is the backend running?`,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const envelope = payload as
      | { error?: { code?: string; message?: string }; detail?: unknown }
      | null;
    const code = envelope?.error?.code ?? "request_failed";
    const message =
      envelope?.error?.message ??
      (typeof envelope?.detail === "string"
        ? envelope.detail
        : `Request failed with status ${response.status}.`);
    throw new ApiError(response.status, code, message);
  }

  return payload as T;
}

export function listDocuments(apiKey: string, signal?: AbortSignal) {
  return request<DocumentSummary[]>("/api/v1/documents", { apiKey, signal });
}

export function getDocument(apiKey: string, id: string, signal?: AbortSignal) {
  return request<DocumentDetail>(`/api/v1/documents/${encodeURIComponent(id)}`, {
    apiKey,
    signal,
  });
}

export function uploadDocument(
  apiKey: string,
  file: File,
  collectionId: string,
): Promise<DocumentDetail> {
  const form = new FormData();
  form.append("file", file);
  form.append("collection_id", collectionId);
  return request<DocumentDetail>("/api/v1/documents", {
    apiKey,
    method: "POST",
    form,
  });
}

export function deleteDocument(apiKey: string, id: string) {
  return request<void>(`/api/v1/documents/${encodeURIComponent(id)}`, {
    apiKey,
    method: "DELETE",
  });
}

export function search(
  apiKey: string,
  payload: { query: string; top_k: number; collection_id: string },
  signal?: AbortSignal,
) {
  return request<SearchResponse>("/api/v1/search", {
    apiKey,
    method: "POST",
    json: payload,
    signal,
  });
}

export function answer(
  apiKey: string,
  payload: { question: string; top_k: number; collection_id: string },
  signal?: AbortSignal,
) {
  return request<AnswerResponse>("/api/v1/answer", {
    apiKey,
    method: "POST",
    json: payload,
    signal,
  });
}

export function getBenchmark(apiKey: string) {
  return request<BenchmarkInfo>("/api/v1/evaluation/benchmark", { apiKey });
}

export function runRetrievalEvaluation(
  apiKey: string,
  payload: { configurations: RetrievalConfiguration[]; k_values: number[]; persist?: boolean },
) {
  return request<EvaluationRecord>("/api/v1/evaluation/retrieval", {
    apiKey,
    method: "POST",
    json: { persist: true, ...payload },
  });
}

export function runAnswerEvaluation(
  apiKey: string,
  payload: {
    chunk_size: number;
    chunk_overlap: number;
    top_k: number;
    include_llm_judge: boolean;
    embedding_model?: string;
  },
) {
  return request<EvaluationRecord>("/api/v1/evaluation/answers", {
    apiKey,
    method: "POST",
    json: { persist: true, ...payload },
  });
}

export function listEvaluationRuns(apiKey: string) {
  return request<RunSummary[]>("/api/v1/evaluation/runs", { apiKey });
}

export function getEvaluationRun(apiKey: string, id: string) {
  return request<EvaluationRecord>(`/api/v1/evaluation/runs/${encodeURIComponent(id)}`, {
    apiKey,
  });
}
