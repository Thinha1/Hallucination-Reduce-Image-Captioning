"use client";

import { Background, ReactFlow, type Edge, type Node, useEdgesState, useNodesState } from "@xyflow/react";
import { useState } from "react";

import ClipScoreNode from "@/components/nodes/ClipScoreNode";
import DetectionNode from "@/components/nodes/DetectionNode";
import CaptionNode from "@/components/nodes/CaptionNode";
import FinalNode from "@/components/nodes/FinalNode";
import ReasoningNode from "@/components/nodes/ReasoningNode";
import RetrievalBankNode from "@/components/nodes/RetrievalBankNode";
import RetrievalNode from "@/components/nodes/RetrievalNode";
import UploadNode from "@/components/nodes/UploadNode";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

const nodeTypes = {
  upload: UploadNode,
  bank: RetrievalBankNode,
  detect2a: DetectionNode,
  retrieve2b: RetrievalNode,
  caption3: CaptionNode,
  clipscore4: ClipScoreNode,
  reasoning5: ReasoningNode,
  final6: FinalNode,
};

const initialNodes: Node[] = [
  { id: "upload", type: "upload", position: { x: 0, y: 60 }, data: {}, style: { width: 300 } },
  { id: "bank", type: "bank", position: { x: 0, y: 480 }, data: {}, style: { width: 300 } },
  { id: "detect2a", type: "detect2a", position: { x: 340, y: 0 }, data: {}, style: { width: 360 } },
  { id: "retrieve2b", type: "retrieve2b", position: { x: 340, y: 420 }, data: {}, style: { width: 380 } },
  { id: "caption3", type: "caption3", position: { x: 770, y: 120 }, data: {}, style: { width: 400 } },
  { id: "clipscore4", type: "clipscore4", position: { x: 1210, y: 120 }, data: {}, style: { width: 360 } },
  { id: "reasoning5", type: "reasoning5", position: { x: 1610, y: 80 }, data: {}, style: { width: 500 } },
  { id: "final6", type: "final6", position: { x: 2150, y: 160 }, data: {}, style: { width: 360 } },
];

const initialEdges: Edge[] = [
  { id: "e-upload-detect", source: "upload", target: "detect2a" },
  { id: "e-upload-retrieve", source: "upload", target: "retrieve2b" },
  { id: "e-bank-retrieve", source: "bank", target: "retrieve2b", animated: true },
  { id: "e-detect-caption", source: "detect2a", target: "caption3" },
  { id: "e-retrieve-caption", source: "retrieve2b", target: "caption3" },
  { id: "e-caption-clipscore", source: "caption3", target: "clipscore4" },
  { id: "e-clipscore-reasoning", source: "clipscore4", target: "reasoning5" },
  { id: "e-reasoning-final", source: "reasoning5", target: "final6" },
];

export default function PipelinePage() {
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);
  const { jobId, reset, runAll, autoRunMode, setAutoRunMode } = usePipelineStore(
    useShallow((s) => ({
      jobId: s.jobId,
      reset: s.reset,
      runAll: s.runAll,
      autoRunMode: s.autoRunMode,
      setAutoRunMode: s.setAutoRunMode,
    }))
  );
  const [runningAll, setRunningAll] = useState(false);
  const [runAllError, setRunAllError] = useState<string | null>(null);

  const handleRunAll = async () => {
    setRunningAll(true);
    setRunAllError(null);
    try {
      await runAll();
    } catch (err) {
      setRunAllError((err as Error).message);
    } finally {
      setRunningAll(false);
    }
  };

  return (
    <div className="flex h-screen w-full flex-col">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-2">
        <div>
          <h1 className="text-sm font-semibold text-gray-800">
            Pipeline sinh chú thích ảnh giảm ảo giác - CLIP-ViT + YOLO-World + truy hồi câu
          </h1>
        </div>
        <div className="flex items-center gap-2">
          {runAllError && <span className="text-[11px] text-red-600">{runAllError}</span>}
          <div
            className="flex gap-0.5 rounded-md bg-gray-100 p-0.5 text-xs font-medium"
            title={
              autoRunMode
                ? "Sau khi tải ảnh, toàn bộ pipeline sẽ tự chạy."
                : 'Sau khi tải ảnh, bấm "Chạy" ở từng bước (hoặc "Chạy tất cả").'
            }
          >
            <button
              type="button"
              onClick={() => setAutoRunMode(true)}
              className={`rounded px-2 py-1 ${autoRunMode ? "bg-white text-indigo-600 shadow-sm" : "text-gray-500"}`}
            >
              Tự động
            </button>
            <button
              type="button"
              onClick={() => setAutoRunMode(false)}
              className={`rounded px-2 py-1 ${!autoRunMode ? "bg-white text-indigo-600 shadow-sm" : "text-gray-500"}`}
            >
              Thủ công
            </button>
          </div>
          <button
            onClick={handleRunAll}
            disabled={!jobId || runningAll}
            className="rounded-md bg-gray-900 px-3 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-300 hover:bg-black"
          >
            {runningAll ? "Đang chạy toàn bộ..." : "Chạy tất cả"}
          </button>
          <button
            onClick={reset}
            className="rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50"
          >
            Làm mới
          </button>
        </div>
      </header>

      <div className="relative min-h-0 flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background />
        </ReactFlow>
      </div>
    </div>
  );
}
