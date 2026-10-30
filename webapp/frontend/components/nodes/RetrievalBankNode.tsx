"use client";

import { useEffect } from "react";

import NodeShell from "../NodeShell";
import { usePipelineStore } from "@/lib/store";
import type { StepStatus } from "@/lib/types";
import { useShallow } from "zustand/react/shallow";

// Shared resource, not a per-image step: the train/restval caption bank encoded with CLIP
// ViT-B/32. Loaded once per backend process; every job's retrieval step reads from it.
export default function RetrievalBankNode({ selected }: { selected?: boolean }) {
  const { bank, bankError, fetchBank, loadBank } = usePipelineStore(
    useShallow((s) => ({
      bank: s.bank,
      bankError: s.bankError,
      fetchBank: s.fetchBank,
      loadBank: s.loadBank,
    }))
  );

  useEffect(() => {
    fetchBank();
  }, [fetchBank]);

  const status: StepStatus = bank?.loading
    ? "running"
    : bank?.loaded
      ? "completed"
      : bankError || bank?.error
        ? "failed"
        : "idle";

  return (
    <NodeShell
      title="Kho truy hồi"
      subtitle={`Câu chú thích tập huấn luyện · ${bank?.model_name ?? "CLIP ViT-B/32"}`}
      status={status}
      onRun={() => loadBank().catch(() => {})}
      runLabel={bank?.loaded ? "Đã nạp" : "Nạp kho"}
      runDisabled={!bank || bank.loaded}
      hasTarget={false}
      width={300}
      selected={selected}
    >
      {!bank ? (
        <p className="text-gray-400">Đang kết nối backend...</p>
      ) : (
        <div className="space-y-2">
          <dl className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-gray-600">
            <dt className="text-gray-400">Số câu</dt>
            <dd className="font-mono">{bank.num_captions?.toLocaleString("vi-VN") ?? "chưa nạp"}</dd>
            <dt className="text-gray-400">Embedding</dt>
            <dd className="font-mono">
              {bank.embedding_dim ? `${bank.embedding_dim} chiều` : "-"}
              {bank.size_mb ? ` · ${bank.size_mb} MB` : ""}
            </dd>
            <dt className="text-gray-400">Truy hồi</dt>
            <dd>
              K = {bank.k}, ngưỡng r = {bank.min_votes}
            </dd>
            <dt className="text-gray-400">Thiết bị</dt>
            <dd className="font-mono">{bank.device}</dd>
            {bank.load_ms !== null && (
              <>
                <dt className="text-gray-400">Thời gian nạp</dt>
                <dd className="font-mono">{(bank.load_ms / 1000).toFixed(1)}s</dd>
              </>
            )}
          </dl>
          {!bank.loaded && (
            <p className="text-[10px] text-gray-400">
              Cần nạp kho trước khi truy hồi câu. &quot;Chạy tất cả&quot; sẽ tự nạp nếu chưa có.
            </p>
          )}
        </div>
      )}
      {(bankError || bank?.error) && <p className="mt-2 text-[11px] text-red-600">{bankError || bank?.error}</p>}
    </NodeShell>
  );
}
