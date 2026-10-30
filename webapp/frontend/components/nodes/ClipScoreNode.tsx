"use client";

import NodeShell from "../NodeShell";
import { clipScoreColor } from "@/lib/colors";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

export default function ClipScoreNode({ selected }: { selected?: boolean }) {
  const { scores, steps, isReady, runStep } = usePipelineStore(
    useShallow((s) => ({
      scores: s.scores,
      steps: s.steps,
      isReady: s.isReady,
      runStep: s.runStep,
    }))
  );

  return (
    <NodeShell
      title="4. Độ khớp ảnh - chữ"
      subtitle="Bộ đánh giá CLIPScore"
      status={steps.clipscore.status}
      onRun={() => runStep("clipscore").catch(() => {})}
      runDisabled={!isReady("clipscore")}
      width={320}
      selected={selected}
    >
      {!scores ? (
        <p className="text-gray-400">Chưa chạy. Bấm Chạy để tính CLIPScore.</p>
      ) : (
        <div className="space-y-2">
          {scores.map((s) => (
            <div key={s.candidate_id}>
              <div className="mb-0.5 flex items-center justify-between gap-2">
                <span className="truncate font-medium text-slate-800" title={s.text}>
                  #{s.candidate_id} {s.text}
                </span>
                <span className="font-mono text-gray-500">{s.clip_score.toFixed(3)}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-gray-100">
                <div
                  className="h-1.5 rounded-full transition-all"
                  style={{ width: `${s.percent}%`, backgroundColor: clipScoreColor(s.clip_score) }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
      {steps.clipscore.status === "failed" && (
        <p className="mt-2 text-[11px] text-red-600">{steps.clipscore.error}</p>
      )}
    </NodeShell>
  );
}
