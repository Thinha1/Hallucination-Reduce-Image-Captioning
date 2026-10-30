import type {
  CaptionStepResponse,
  ClipScoreStepResponse,
  DetectStepResponse,
  FinalStepResponse,
  ModelsResponse,
  RerankStepResponse,
  RetrievalStatus,
  RetrieveStepResponse,
  UploadResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const API_HEADERS = {
  "ngrok-skip-browser-warning": "true",
};

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function absoluteImageUrl(imageUrl: string): string {
  if (imageUrl.startsWith("http") || imageUrl.startsWith("data:")) return imageUrl;
  return `${API_BASE}${imageUrl}`;
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${API_BASE}/api/jobs`, { method: "POST", headers: API_HEADERS, body: form });
  return handle<UploadResponse>(res);
}

async function post<T>(path: string, signal?: AbortSignal, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: body !== undefined ? { ...API_HEADERS, "Content-Type": "application/json" } : API_HEADERS,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });
  return handle<T>(res);
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: API_HEADERS, signal });
  return handle<T>(res);
}

export const getModels = (signal?: AbortSignal) => get<ModelsResponse>(`/api/models`, signal);

export const getRetrievalStatus = (signal?: AbortSignal) => get<RetrievalStatus>(`/api/retrieval`, signal);

export const loadRetrievalBank = (signal?: AbortSignal) => post<RetrievalStatus>(`/api/retrieval/load`, signal);

export const runRetrieveStep = (jobId: string, signal?: AbortSignal) =>
  post<RetrieveStepResponse>(`/api/jobs/${jobId}/steps/retrieve`, signal);

// delta = null lets the backend use the model's default from weights/models.json.
export const runCaptionStep = (jobId: string, modelId: string | null, delta: number | null, signal?: AbortSignal) =>
  post<CaptionStepResponse>(`/api/jobs/${jobId}/steps/caption`, signal, { model_id: modelId, delta });

export const runDetectStep = (jobId: string, signal?: AbortSignal) =>
  post<DetectStepResponse>(`/api/jobs/${jobId}/steps/detect`, signal);

export const runClipScoreStep = (jobId: string, signal?: AbortSignal) =>
  post<ClipScoreStepResponse>(`/api/jobs/${jobId}/steps/clipscore`, signal);

export const runRerankStep = (jobId: string, signal?: AbortSignal) =>
  post<RerankStepResponse>(`/api/jobs/${jobId}/steps/rerank`, signal);

export const getFinalStep = (jobId: string, signal?: AbortSignal) =>
  get<FinalStepResponse>(`/api/jobs/${jobId}/steps/final`, signal);
