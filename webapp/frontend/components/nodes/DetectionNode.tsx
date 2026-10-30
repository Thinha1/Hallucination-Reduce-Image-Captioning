"use client";

import NodeShell from "../NodeShell";
import BBoxOverlay from "../BBoxOverlay";
import { labelColor } from "@/lib/colors";
import { usePipelineStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

export default function DetectionNode({ selected }: { selected?: boolean }) {
  const { imageUrl, imageMeta, objects, steps, isReady, runStep } = usePipelineStore(
    useShallow((s) => ({
      imageUrl: s.imageUrl,
      imageMeta: s.imageMeta,
      objects: s.objects,
      steps: s.steps,
      isReady: s.isReady,
      runStep: s.runStep,
    }))
  );

  const uniqueLabels = objects ? Array.from(new Set(objects.map((o) => o.label))) : [];

  return (
    <NodeShell
      title="2A. Nhận diện vật thể"
      subtitle="YOLO-World"
      status={steps.detect.status}
      onRun={() => runStep("detect").catch(() => {})}
      runDisabled={!isReady("detect")}
      width={320}
      selected={selected}
    >
      {!objects || !imageUrl || !imageMeta ? (
        <p className="text-gray-400">Chưa chạy. Bấm Chạy để nhận diện vật thể.</p>
      ) : (
        <div className="space-y-2">
          <BBoxOverlay
            imageUrl={imageUrl}
            objects={objects}
            naturalWidth={imageMeta.width}
            naturalHeight={imageMeta.height}
          />
          <div className="flex flex-wrap gap-1">
            {uniqueLabels.map((label) => (
              <span
                key={label}
                className="rounded-full px-2 py-0.5 text-[10px] font-medium text-white"
                style={{ backgroundColor: labelColor(label) }}
              >
                {label}
              </span>
            ))}
            {uniqueLabels.length === 0 && <span className="text-gray-400">Không phát hiện vật thể nào.</span>}
          </div>
        </div>
      )}
      {steps.detect.status === "failed" && (
        <p className="mt-2 text-[11px] text-red-600">{steps.detect.error}</p>
      )}
    </NodeShell>
  );
}
