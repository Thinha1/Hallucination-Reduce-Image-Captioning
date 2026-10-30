"use client";

import type { StepStatus } from "@/lib/types";

const LABELS: Record<StepStatus, string> = {
  idle: "Chưa chạy",
  running: "Đang chạy...",
  completed: "Hoàn tất",
  failed: "Lỗi",
};

const STYLES: Record<StepStatus, string> = {
  idle: "bg-gray-100 text-gray-500 border-gray-200",
  running: "bg-blue-50 text-blue-700 border-blue-300 animate-pulse",
  completed: "bg-green-50 text-green-700 border-green-300",
  failed: "bg-red-50 text-red-700 border-red-300",
};

export default function StatusBadge({ status }: { status: StepStatus }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${STYLES[status]}`}>
      {status === "running" && (
        <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-ping" />
      )}
      {LABELS[status]}
    </span>
  );
}
