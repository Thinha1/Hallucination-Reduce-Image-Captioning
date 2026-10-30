"use client";

import NodeShell from "../NodeShell";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

export default function FinalNode({ selected }: { selected?: boolean }) {
  const { final, steps, isReady, runStep } = usePipelineStore(
    useShallow((s) => ({
      final: s.final,
      steps: s.steps,
      isReady: s.isReady,
      runStep: s.runStep,
    }))
  );

  return (
    <NodeShell
      title="6. Kết quả cuối"
      subtitle="Chú thích được chọn"
      status={steps.final.status}
      onRun={() => runStep("final").catch(() => {})}
      runDisabled={!isReady("final")}
      runLabel="Xem kết quả"
      hasSource={false}
      width={320}
      selected={selected}
    >
      {!final ? (
        <p className="text-gray-400">Chưa có kết quả. Hoàn tất Bước 5 rồi bấm nút bên dưới.</p>
      ) : (
        <div className="space-y-3">
          <div className="rounded-lg border-2 border-amber-300 bg-amber-50 p-3 text-center shadow-inner">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-amber-700">
              Chú thích cuối
            </div>
            <p className="text-sm font-semibold leading-relaxed text-slate-900">&ldquo;{final.final_caption}&rdquo;</p>
          </div>

          <dl className="space-y-1 text-gray-600">
            <div className="flex items-center justify-between">
              <dt className="text-gray-400">CLIPScore cuối</dt>
              <dd className="font-mono">{final.final_clip_score?.toFixed(3) ?? "N/A"}</dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-gray-400">Trạng thái ảo giác</dt>
              <dd className={final.hallucination_violations === 0 ? "text-green-600" : "text-red-600"}>
                {final.hallucination_status}
              </dd>
            </div>
            <div className="flex items-center justify-between">
              <dt className="text-gray-400">Tổng thời gian pipeline</dt>
              <dd className="font-mono">{(final.total_latency_ms / 1000).toFixed(1)}s</dd>
            </div>
          </dl>
        </div>
      )}
      {steps.final.status === "failed" && (
        <p className="mt-2 text-[11px] text-red-600">{steps.final.error}</p>
      )}
    </NodeShell>
  );
}
