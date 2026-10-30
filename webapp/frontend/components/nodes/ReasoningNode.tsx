"use client";

import type { ReactNode } from "react";

import NodeShell from "../NodeShell";
import { usePipelineStore } from "@/lib/store";
import type { FormulaBreakdownOut, HallucinationOut } from "@/lib/types";
import { useShallow } from "zustand/react/shallow";

type SignalKey = "s_base" | "s_align" | "s_belief" | "s_obj" | "s_retr";

const BREAKDOWN_TERMS: { key: SignalKey; label: string; hint: string }[] = [
  { key: "s_base", label: "Base", hint: "độ tin cậy của mô hình sinh" },
  { key: "s_align", label: "Align", hint: "CLIPScore ảnh - câu" },
  { key: "s_belief", label: "Belief", hint: "belief revision theo YOLO" },
  { key: "s_obj", label: "Obj", hint: "phạt vật thể ngoài bằng chứng E(I)" },
  { key: "s_retr", label: "Retr", hint: "giống các câu truy hồi" },
];

// Breakdown values are already min-max normalized to [0, 1] across the candidate set (see
// FormulaReranker._normalize in the backend) -- a plain low/mid/high traffic-light scale, not
// the raw-CLIPScore-calibrated thresholds in lib/colors.ts.
function breakdownColor(value: number): string {
  if (value >= 0.66) return "#16a34a";
  if (value >= 0.33) return "#ca8a04";
  return "#dc2626";
}

function FormulaBreakdownBars({ breakdown }: { breakdown: FormulaBreakdownOut }) {
  const terms = BREAKDOWN_TERMS.filter(({ key }) => breakdown[key] !== null && breakdown[key] !== undefined);
  return (
    <div className="mt-1 grid gap-1.5" style={{ gridTemplateColumns: `repeat(${terms.length}, minmax(0, 1fr))` }}>
      {terms.map(({ key, label, hint }) => {
        const value = breakdown[key] as number;
        return (
          <div key={key} title={`S_${key.slice(2)} = ${value.toFixed(3)} (đã chuẩn hoá 0-1): ${hint}`}>
            <div className="flex items-center justify-between text-[9px] text-gray-400">
              <span>{label}</span>
              <span className="font-mono">{value.toFixed(2)}</span>
            </div>
            <div className="h-1 w-full rounded-full bg-gray-100">
              <div
                className="h-1 rounded-full transition-all"
                style={{ width: `${value * 100}%`, backgroundColor: breakdownColor(value) }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function renderHighlighted(text: string, flags: HallucinationOut[]): ReactNode[] {
  if (flags.length === 0) return [text];
  const flagByWord = new Map<string, HallucinationOut>();
  flags.forEach((f) => flagByWord.set(f.word.toLowerCase(), f));

  return text.split(/(\s+)/).map((part, idx) => {
    const clean = part.toLowerCase().replace(/[.,!?;:]/g, "");
    const flag = flagByWord.get(clean);
    if (!flag) return <span key={idx}>{part}</span>;
    const color = "#dc2626";
    return (
      <mark
        key={idx}
        title={flag.note}
        style={{ backgroundColor: "transparent", color, textDecoration: `underline wavy ${color}`, fontWeight: 600 }}
      >
        {part}
        <span aria-hidden> ⚠️</span>
      </mark>
    );
  });
}

export default function ReasoningNode({ selected }: { selected?: boolean }) {
  const { rerank, steps, isReady, runStep } = usePipelineStore(
    useShallow((s) => ({
      rerank: s.rerank,
      steps: s.steps,
      isReady: s.isReady,
      runStep: s.runStep,
    }))
  );

  return (
    <NodeShell
      title="5. Chọn lại câu và kiểm tra ảo giác"
      subtitle="Công thức năm tín hiệu"
      status={steps.rerank.status}
      onRun={() => runStep("rerank").catch(() => {})}
      runDisabled={!isReady("rerank")}
      width={460}
      selected={selected}
    >
      <div className="space-y-3">
        <p className="text-[10px] text-gray-400">
          S_base (độ tin cậy của mô hình), S_align (CLIPScore), S_belief (belief revision theo YOLO), S_obj (phạt vật
          thể ngoài bằng chứng E(I)) và S_retr (giống các câu truy hồi), chuẩn hoá min-max trong từng ảnh.
        </p>

        {!rerank ? (
          <p className="text-gray-400">
            Chưa chạy. Bước này chọn chú thích tốt nhất trong các ứng viên và đánh dấu từ vật thể không có
            trong bằng chứng.
          </p>
        ) : (
          <div className="space-y-1.5">
            <p className="text-[10px] text-gray-400">{rerank.model_name}</p>
            {rerank.ranking.map((item) => {
              const flags = rerank.hallucinations.filter((h) => h.candidate_id === item.candidate_id);
              return (
                <div
                  key={item.candidate_id}
                  className={`rounded-md border px-2 py-1 ${item.rank === 1 ? "border-indigo-300 bg-indigo-50" : "border-gray-100 bg-white"}`}
                >
                  <div className="flex items-center justify-between text-[10px] text-gray-400">
                    <span>#{item.candidate_id} - rank {item.rank}</span>
                    <span className="font-mono">{item.score.toFixed(3)}</span>
                  </div>
                  <div className="font-medium text-slate-800">{renderHighlighted(item.text, flags)}</div>
                  {item.formula_breakdown && <FormulaBreakdownBars breakdown={item.formula_breakdown} />}
                </div>
              );
            })}
          </div>
        )}
      </div>
      {steps.rerank.status === "failed" && (
        <p className="mt-2 text-[11px] text-red-600">{steps.rerank.error}</p>
      )}
    </NodeShell>
  );
}
