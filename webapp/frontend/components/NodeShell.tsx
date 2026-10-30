"use client";

import { Handle, NodeResizer, Position } from "@xyflow/react";
import type { ReactNode } from "react";

import StatusBadge from "./StatusBadge";
import type { StepStatus } from "@/lib/types";

interface Props {
  title: string;
  subtitle?: string;
  status: StepStatus;
  onRun?: () => void;
  runLabel?: string;
  runDisabled?: boolean;
  hasTarget?: boolean;
  hasSource?: boolean;
  width?: number;
  selected?: boolean;
  children: ReactNode;
}

export default function NodeShell({
  title,
  subtitle,
  status,
  onRun,
  runLabel = "Chạy",
  runDisabled,
  hasTarget = true,
  hasSource = true,
  width = 320,
  selected = false,
  children,
}: Props) {
  return (
    <div
      className="flex h-full min-h-[120px] w-full flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm"
      style={{ minWidth: Math.min(width, 240) }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={240}
        minHeight={120}
        handleClassName="!h-2.5 !w-2.5 !border-indigo-600 !bg-white"
        lineClassName="!border-indigo-500"
      />
      {hasTarget && <Handle type="target" position={Position.Left} className="!bg-gray-400" />}
      {hasSource && <Handle type="source" position={Position.Right} className="!bg-gray-400" />}

      <div className="flex items-start justify-between gap-2 rounded-t-xl border-b border-gray-100 bg-gray-50 px-3 py-2">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-gray-800">{title}</div>
          {subtitle && <div className="truncate text-[11px] text-gray-500">{subtitle}</div>}
        </div>
        <StatusBadge status={status} />
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-3 text-xs">{children}</div>

      {onRun && (
        <div className="border-t border-gray-100 px-3 py-2">
          <button
            onClick={onRun}
            disabled={runDisabled || status === "running"}
            className="nodrag w-full rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white transition disabled:cursor-not-allowed disabled:bg-gray-300 hover:bg-indigo-700"
          >
            {status === "running" ? "Đang chạy..." : runLabel}
          </button>
        </div>
      )}
    </div>
  );
}
