"use client";

import { useRef } from "react";

import NodeShell from "../NodeShell";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadNode({ selected }: { selected?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const { imageUrl, imageMeta, steps, upload, runAll, autoRunMode } = usePipelineStore(
    useShallow((s) => ({
      imageUrl: s.imageUrl,
      imageMeta: s.imageMeta,
      steps: s.steps,
      upload: s.upload,
      runAll: s.runAll,
      autoRunMode: s.autoRunMode,
    }))
  );

  const handleFile = async (file: File | null) => {
    if (!file) return;
    try {
      await upload(file);
      if (autoRunMode) await runAll();
    } catch {
      // status is already recorded as "failed" by the store; nothing else to do here.
    }
  };

  return (
    <NodeShell
      title="1. Ảnh đầu vào"
      subtitle="Tải ảnh và tiền xử lý"
      status={steps.upload.status}
      hasTarget={false}
      width={280}
      selected={selected}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
      />

      {!imageUrl ? (
        <button
          onClick={() => inputRef.current?.click()}
          className="flex w-full flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed border-gray-300 py-6 text-gray-500 hover:border-indigo-400 hover:text-indigo-500"
        >
          <span className="text-2xl">+</span>
          <span>{autoRunMode ? "Chọn ảnh để tự chạy pipeline" : "Chọn ảnh để bắt đầu"}</span>
        </button>
      ) : (
        <div className="space-y-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imageUrl} alt="Ảnh đã tải lên" className="mx-auto max-h-40 rounded-md border border-gray-100 object-contain" />
          <dl className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-gray-600">
            <dt className="text-gray-400">Kích thước</dt>
            <dd>{imageMeta?.width}×{imageMeta?.height}</dd>
            <dt className="text-gray-400">Định dạng</dt>
            <dd>{imageMeta?.format}</dd>
            <dt className="text-gray-400">Dung lượng</dt>
            <dd>{imageMeta ? formatBytes(imageMeta.size_bytes) : "-"}</dd>
          </dl>
          <button
            onClick={() => inputRef.current?.click()}
            className="w-full rounded-md border border-gray-200 py-1 text-[11px] text-gray-500 hover:bg-gray-50"
          >
            Đổi ảnh khác
          </button>
        </div>
      )}
      {steps.upload.status === "failed" && (
        <p className="mt-2 text-[11px] text-red-600">{steps.upload.error}</p>
      )}
    </NodeShell>
  );
}
