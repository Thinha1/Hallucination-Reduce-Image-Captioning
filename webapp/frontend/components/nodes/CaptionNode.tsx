"use client";

import { useEffect, useState } from "react";

import NodeShell from "../NodeShell";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

export default function CaptionNode({ selected }: { selected?: boolean }) {
  const { candidates, captionInfo, steps, isReady, runStep, models, selectedModelId, delta, fetchModels, setModel, setDelta } =
    usePipelineStore(
      useShallow((s) => ({
        candidates: s.candidates,
        captionInfo: s.captionInfo,
        steps: s.steps,
        isReady: s.isReady,
        runStep: s.runStep,
        models: s.models,
        selectedModelId: s.selectedModelId,
        delta: s.delta,
        fetchModels: s.fetchModels,
        setModel: s.setModel,
        setDelta: s.setDelta,
      }))
    );
  const [showBlocked, setShowBlocked] = useState(false);

  useEffect(() => {
    fetchModels().catch(() => {});
  }, [fetchModels]);

  const model = models.find((m) => m.id === selectedModelId);
  const effectiveDelta = delta ?? model?.default_delta ?? 0;
  const running = steps.caption.status === "running";

  return (
    <NodeShell
      title="3. Sinh chú thích"
      subtitle="CLIP-ViT + Transformer Decoder, chặn logits theo bằng chứng"
      status={steps.caption.status}
      onRun={() => runStep("caption").catch(() => {})}
      runDisabled={!isReady("caption")}
      width={380}
      selected={selected}
    >
      <div className="space-y-3">
        <div className="space-y-1.5">
          <label className="block">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Mô hình</span>
            <select
              value={selectedModelId ?? ""}
              onChange={(e) => setModel(e.target.value)}
              disabled={running || models.length === 0}
              className="nodrag mt-0.5 w-full rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-slate-700"
            >
              {models.length === 0 && <option value="">Đang tải danh sách...</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                  {m.loaded ? "" : " (nạp khi chạy)"}
                </option>
              ))}
            </select>
          </label>
          {model && <p className="text-[10px] text-gray-400">{model.description}</p>}
          <label className="flex items-center justify-between gap-2">
            <span className="text-[10px] text-gray-500" title="z' = z − δ cho từ vật thể không có trong bằng chứng E(I). δ = 0 là không chặn.">
              Mức chặn logits δ
            </span>
            <input
              type="number"
              min={0}
              step={0.5}
              value={effectiveDelta}
              onChange={(e) => setDelta(e.target.value === "" ? null : Math.max(0, Number(e.target.value)))}
              disabled={running}
              className="nodrag w-16 rounded-md border border-gray-200 px-1.5 py-0.5 text-right font-mono text-xs"
            />
          </label>
        </div>

        {captionInfo && (
          <div className="space-y-1 rounded-md bg-gray-50 p-2">
            <div className="text-[10px] text-gray-500">
              Đã chạy: <span className="font-medium text-slate-700">{captionInfo.model_label}</span>, δ ={" "}
              <span className="font-mono">{captionInfo.delta}</span>
            </div>
            <div className="flex flex-wrap items-center gap-1">
              <span className="text-[10px] text-gray-400">Bằng chứng E(I):</span>
              {captionInfo.evidence.objects.map((o) => (
                <span
                  key={o}
                  title={[
                    captionInfo.evidence.detected.includes(o) ? "YOLO" : null,
                    captionInfo.evidence.retrieved.includes(o) ? "câu truy hồi" : null,
                  ]
                    .filter(Boolean)
                    .join(" + ")}
                  className="rounded-full border border-green-300 bg-white px-1.5 py-0.5 text-[10px] text-green-700"
                >
                  {o}
                </span>
              ))}
              {captionInfo.evidence.objects.length === 0 && <span className="text-[10px] text-gray-400">trống</span>}
            </div>
            {captionInfo.blocked_objects.length > 0 && (
              <div className="text-[10px] text-gray-500">
                Bị chặn {captionInfo.blocked_objects.length} vật thể COCO khác.{" "}
                <button type="button" onClick={() => setShowBlocked((v) => !v)} className="nodrag text-indigo-600 hover:underline">
                  {showBlocked ? "Ẩn" : "Xem"}
                </button>
                {showBlocked && <p className="mt-0.5 text-gray-400">{captionInfo.blocked_objects.join(", ")}</p>}
              </div>
            )}
          </div>
        )}

        {!candidates ? (
          <p className="text-gray-400">Chưa chạy. Cần nhận diện vật thể và truy hồi câu trước, rồi bấm Chạy.</p>
        ) : (
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-left text-gray-400">
                <th className="pb-1 pr-1">#</th>
                <th className="pb-1">Chú thích</th>
                <th className="pb-1 pl-1 text-right">Log Prob</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr key={c.id} className="border-t border-gray-100 align-top">
                  <td className="py-1 pr-1 text-gray-400">{c.id}</td>
                  <td className="py-1 pr-1 font-medium text-slate-800">
                    {c.text}
                    {c.source.endsWith(":safe") && (
                      <span className="ml-1 rounded bg-gray-100 px-1 text-[9px] font-normal text-gray-500">beam</span>
                    )}
                  </td>
                  <td className="py-1 pl-1 text-right font-mono text-gray-500">
                    {c.generation_score !== null ? c.generation_score.toFixed(2) : "N/A"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {steps.caption.status === "failed" && <p className="mt-2 text-[11px] text-red-600">{steps.caption.error}</p>}
    </NodeShell>
  );
}
