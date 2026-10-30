// Mirrors webapp/backend/models.py (the API's pydantic response shapes).

export type StepKey = "upload" | "detect" | "retrieve" | "caption" | "clipscore" | "rerank" | "final";
export type StepStatus = "idle" | "running" | "completed" | "failed";

export interface UploadResponse {
  job_id: string;
  image_url: string;
  image_data_url?: string;
  width: number;
  height: number;
  format: string;
  size_bytes: number;
  filename: string;
  status: "completed";
}

export interface CandidateOut {
  id: number;
  text: string;
  source: string;
  generation_score: number | null;
  clip_score: number | null;
}

export interface CaptionModelOut {
  id: string;
  label: string;
  description: string;
  default_delta: number;
  loaded: boolean;
}

export interface ModelsResponse {
  models: CaptionModelOut[];
  default_model_id: string;
}

export interface RetrievalStatus {
  loaded: boolean;
  loading: boolean;
  num_captions: number | null;
  embedding_dim: number | null;
  size_mb: number | null;
  model_name: string;
  k: number;
  min_votes: number;
  device: string;
  load_ms: number | null;
  error: string | null;
}

export interface RetrievedCaptionOut {
  index: number;
  caption: string;
  similarity: number;
  objects: string[];
}

export interface ObjectVoteOut {
  object: string;
  votes: number;
  in_evidence: boolean;
}

export interface RetrieveStepResponse {
  neighbours: RetrievedCaptionOut[];
  votes: ObjectVoteOut[];
  k: number;
  min_votes: number;
  latency_ms: number;
}

export interface EvidenceOut {
  detected: string[];
  retrieved: string[];
  objects: string[];
}

export interface CaptionStepResponse {
  candidates: CandidateOut[];
  model_id: string;
  model_label: string;
  delta: number;
  evidence: EvidenceOut;
  blocked_objects: string[];
  latency_ms: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObjectOut {
  label: string;
  confidence: number;
  box: BoundingBox | null;
}

export interface DetectStepResponse {
  objects: DetectedObjectOut[];
  latency_ms: number;
}

export interface ClipScoreOut {
  candidate_id: number;
  text: string;
  clip_score: number;
  percent: number;
}

export interface ClipScoreStepResponse {
  scores: ClipScoreOut[];
  latency_ms: number;
}

export interface FormulaBreakdownOut {
  s_base: number;
  s_align: number;
  s_belief: number;
  // Retrieval signals of the proposed method; null when the job ran without them.
  s_obj: number | null;
  s_retr: number | null;
  formula_score: number | null;
}

export interface RankingOut {
  candidate_id: number;
  rank: number;
  score: number;
  text: string;
  formula_breakdown: FormulaBreakdownOut | null;
}

export interface HallucinationOut {
  candidate_id: number;
  word: string;
  note: string;
}

export interface RerankStepResponse {
  ranking: RankingOut[];
  selected_caption: string | null;
  hallucinations: HallucinationOut[];
  model_name: string;
  // True when S_obj and S_retr were added on top of the three-signal formula.
  retrieval_signals: boolean;
  latency_ms: number;
}

export interface FinalStepResponse {
  final_caption: string;
  final_clip_score: number | null;
  hallucination_violations: number;
  hallucination_status: string;
  total_latency_ms: number;
}

export interface ApiErrorBody {
  detail: string;
}
