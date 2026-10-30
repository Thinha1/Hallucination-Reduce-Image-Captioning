import { create } from "zustand";

import {
  absoluteImageUrl,
  getFinalStep,
  getModels,
  getRetrievalStatus,
  loadRetrievalBank,
  runCaptionStep,
  runClipScoreStep,
  runDetectStep,
  runRerankStep,
  runRetrieveStep,
  uploadImage,
} from "./api";
import type {
  CandidateOut,
  CaptionModelOut,
  CaptionStepResponse,
  ClipScoreOut,
  DetectedObjectOut,
  FinalStepResponse,
  RerankStepResponse,
  RetrievalStatus,
  RetrieveStepResponse,
  StepKey,
  StepStatus,
  UploadResponse,
} from "./types";

// Mirrors webapp/backend/jobs.py::STEP_DEPENDENCIES -- kept in sync by hand since the frontend
// needs it to disable "Run" buttons before ever calling the API.
export const STEP_DEPENDENCIES: Record<StepKey, StepKey[]> = {
  upload: [],
  detect: ["upload"],
  retrieve: ["upload"],
  caption: ["detect", "retrieve"],
  clipscore: ["caption"],
  rerank: ["detect", "retrieve", "clipscore"],
  final: ["rerank"],
};

interface StepState {
  status: StepStatus;
  latencyMs?: number;
  error?: string;
}

const idleSteps = (): Record<StepKey, StepState> => ({
  upload: { status: "idle" },
  detect: { status: "idle" },
  retrieve: { status: "idle" },
  caption: { status: "idle" },
  clipscore: { status: "idle" },
  rerank: { status: "idle" },
  final: { status: "idle" },
});

const emptyJob = () => ({
  jobId: null,
  imageUrl: null,
  imageMeta: null,
  steps: idleSteps(),
  objects: null,
  retrieval: null,
  captionInfo: null,
  candidates: null,
  scores: null,
  rerank: null,
  final: null,
});

interface PipelineState {
  jobId: string | null;
  imageUrl: string | null;
  imageMeta: UploadResponse | null;
  steps: Record<StepKey, StepState>;

  objects: DetectedObjectOut[] | null;
  retrieval: RetrieveStepResponse | null;
  // Model, logit-blocking strength, evidence E(I) and blocked objects of the last caption run.
  captionInfo: Omit<CaptionStepResponse, "candidates" | "latency_ms"> | null;
  candidates: CandidateOut[] | null;
  scores: ClipScoreOut[] | null;
  rerank: RerankStepResponse | null;
  final: FinalStepResponse | null;

  // Caption checkpoints listed in backend/weights/models.json, and the user's choice. delta =
  // null means "use the selected model's default".
  models: CaptionModelOut[];
  selectedModelId: string | null;
  delta: number | null;

  // The retrieval bank is shared by all jobs and loaded on demand from its own node.
  bank: RetrievalStatus | null;
  bankError: string | null;

  // Whether the pipeline runs all steps automatically right after upload ("auto", the
  // long-standing default) or waits for the user to press "Run" on each node ("manual").
  autoRunMode: boolean;

  reset: () => void;
  upload: (file: File) => Promise<void>;
  setAutoRunMode: (auto: boolean) => void;
  fetchModels: () => Promise<void>;
  setModel: (modelId: string) => void;
  setDelta: (delta: number | null) => void;
  fetchBank: () => Promise<void>;
  loadBank: () => Promise<void>;
  runStep: (step: Exclude<StepKey, "upload">) => Promise<void>;
  runAll: () => Promise<void>;
  isReady: (step: StepKey) => boolean;
}

// Steps whose results become stale when `step` is re-run.
const DOWNSTREAM: Partial<Record<StepKey, StepKey[]>> = {
  caption: ["clipscore", "rerank", "final"],
  clipscore: ["rerank", "final"],
  rerank: ["final"],
};

export const usePipelineStore = create<PipelineState>((set, get) => ({
  ...emptyJob(),
  models: [],
  selectedModelId: null,
  delta: null,
  bank: null,
  bankError: null,
  autoRunMode: true,

  // Note: model/delta/bank/autoRunMode are intentionally left out of reset()/upload()'s
  // cleared fields -- they are settings or shared resources, not per-job state.
  reset: () => set(emptyJob()),

  isReady: (step) => {
    const { steps } = get();
    return STEP_DEPENDENCIES[step].every((dep) => steps[dep].status === "completed");
  },

  upload: async (file) => {
    set(emptyJob());
    try {
      const res = await uploadImage(file);
      set((state) => ({
        jobId: res.job_id,
        imageUrl: res.image_data_url ?? absoluteImageUrl(res.image_url),
        imageMeta: res,
        steps: { ...state.steps, upload: { status: "completed" } },
      }));
    } catch (err) {
      set((state) => ({
        steps: { ...state.steps, upload: { status: "failed", error: (err as Error).message } },
      }));
      throw err;
    }
  },

  setAutoRunMode: (auto) => set({ autoRunMode: auto }),

  fetchModels: async () => {
    const res = await getModels();
    set((state) => ({
      models: res.models,
      selectedModelId: state.selectedModelId ?? res.default_model_id,
    }));
  },
  setModel: (modelId) => set({ selectedModelId: modelId, delta: null }),
  setDelta: (delta) => set({ delta }),

  fetchBank: async () => {
    try {
      set({ bank: await getRetrievalStatus(), bankError: null });
    } catch (err) {
      set({ bankError: (err as Error).message });
    }
  },

  loadBank: async () => {
    set((state) => ({ bank: state.bank ? { ...state.bank, loading: true } : state.bank, bankError: null }));
    try {
      set({ bank: await loadRetrievalBank() });
    } catch (err) {
      set((state) => ({
        bank: state.bank ? { ...state.bank, loading: false } : state.bank,
        bankError: (err as Error).message,
      }));
      throw err;
    }
  },

  runStep: async (step) => {
    const { jobId } = get();
    if (!jobId) throw new Error("Chưa có job nào. Hãy upload ảnh trước.");
    if (!get().isReady(step)) throw new Error(`Bước '${step}' chưa đủ điều kiện để chạy.`);

    set((state) => {
      const steps = { ...state.steps, [step]: { status: "running" as StepStatus } };
      for (const later of DOWNSTREAM[step] ?? []) steps[later] = { status: "idle" };
      return { steps };
    });
    try {
      switch (step) {
        case "detect": {
          const res = await runDetectStep(jobId);
          set((state) => ({
            objects: res.objects,
            steps: { ...state.steps, detect: { status: "completed", latencyMs: res.latency_ms } },
          }));
          break;
        }
        case "retrieve": {
          if (!get().bank?.loaded) await get().loadBank();
          const res = await runRetrieveStep(jobId);
          set((state) => ({
            retrieval: res,
            steps: { ...state.steps, retrieve: { status: "completed", latencyMs: res.latency_ms } },
          }));
          break;
        }
        case "caption": {
          const { selectedModelId, delta } = get();
          const { candidates, latency_ms, ...info } = await runCaptionStep(jobId, selectedModelId, delta);
          set((state) => ({
            candidates,
            captionInfo: info,
            scores: null,
            rerank: null,
            final: null,
            models: state.models.map((m) => (m.id === info.model_id ? { ...m, loaded: true } : m)),
            steps: { ...state.steps, caption: { status: "completed", latencyMs: latency_ms } },
          }));
          break;
        }
        case "clipscore": {
          const res = await runClipScoreStep(jobId);
          set((state) => ({
            scores: res.scores,
            rerank: null,
            final: null,
            steps: { ...state.steps, clipscore: { status: "completed", latencyMs: res.latency_ms } },
          }));
          break;
        }
        case "rerank": {
          const res = await runRerankStep(jobId);
          set((state) => ({
            rerank: res,
            final: null,
            steps: { ...state.steps, rerank: { status: "completed", latencyMs: res.latency_ms } },
          }));
          break;
        }
        case "final": {
          const res = await getFinalStep(jobId);
          set((state) => ({
            final: res,
            steps: { ...state.steps, final: { status: "completed", latencyMs: res.total_latency_ms } },
          }));
          break;
        }
      }
    } catch (err) {
      set((state) => ({
        steps: { ...state.steps, [step]: { status: "failed", error: (err as Error).message } },
      }));
      throw err;
    }
  },

  runAll: async () => {
    await Promise.all([get().runStep("detect"), get().runStep("retrieve")]);
    await get().runStep("caption");
    await get().runStep("clipscore");
    await get().runStep("rerank");
    await get().runStep("final");
  },
}));
