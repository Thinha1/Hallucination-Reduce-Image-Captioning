"use client";

import NodeShell from "../NodeShell";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

export default function RetrievalNode({ selected }: { selected?: boolean }) {
  const { retrieval, bank, steps, isReady, runStep } = usePipelineStore(
    useShallow((s) => ({
      retrieval: s.retrieval,
      bank: s.bank,
      steps: s.steps,
      isReady: s.isReady,
      runStep: s.runStep,
    }))
  );

  return (
    <NodeShell
      title="2B. Truy hồi câu"
      subtitle={`${bank?.k ?? 5} câu gần ảnh nhất trong kho`}
      status={steps.retrieve.status}
      onRun={() => runStep("retrieve").catch(() => {})}
      runDisabled={!isReady("retrieve")}
      width={360}
      selected={selected}
    >
      {!retrieval ? (
        <p className="text-gray-400">
          Chưa chạy. Bấm Chạy để tìm các câu chú thích gần ảnh nhất
          {bank?.loaded ? "." : " (kho sẽ được nạp trước)."}
        </p>
      ) : (
        <div className="space-y-2">
          <ol className="space-y-1">
            {retrieval.neighbours.map((n, i) => (
              <li key={n.index} className="flex items-start justify-between gap-2 border-t border-gray-100 pt-1 first:border-0 first:pt-0">
                <span className="text-slate-700">
                  <span className="mr-1 text-gray-400">{i + 1}.</span>
                  {n.caption}
                </span>
                <span className="font-mono text-gray-500">{n.similarity.toFixed(3)}</span>
              </li>
            ))}
          </ol>
          <div>
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-gray-400">
              Vật thể được nhắc (≥ {retrieval.min_votes}/{retrieval.k} câu thì vào bằng chứng)
            </div>
            <div className="flex flex-wrap gap-1">
              {retrieval.votes.map((v) => (
                <span
                  key={v.object}
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                    v.in_evidence ? "border-green-300 bg-green-50 text-green-700" : "border-gray-200 bg-gray-50 text-gray-500"
                  }`}
                >
                  {v.object} {v.votes}/{retrieval.k}
                </span>
              ))}
              {retrieval.votes.length === 0 && <span className="text-gray-400">Không câu nào nhắc vật thể COCO.</span>}
            </div>
          </div>
        </div>
      )}
      {steps.retrieve.status === "failed" && <p className="mt-2 text-[11px] text-red-600">{steps.retrieve.error}</p>}
    </NodeShell>
  );
}
