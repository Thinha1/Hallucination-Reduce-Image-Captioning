"use client";

import type { DetectedObjectOut } from "@/lib/types";
import { labelColor } from "@/lib/colors";

interface Props {
  imageUrl: string;
  objects: DetectedObjectOut[];
  naturalWidth: number;
  naturalHeight: number;
}

/** Draws colored bounding boxes + label badges over the original image using absolute-positioned
 * divs sized in percent of the container -- no server-side image drawing needed, since
 * DetectedObject.box_xyxy already carries pixel coordinates in the original image's frame. */
export default function BBoxOverlay({ imageUrl, objects, naturalWidth, naturalHeight }: Props) {
  return (
    <div className="relative w-full overflow-hidden rounded-md border border-black/10">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imageUrl} alt="detected objects" className="block w-full h-auto" />
      {objects.map((obj, idx) => {
        if (!obj.box) return null;
        const color = labelColor(obj.label);
        const left = (obj.box.x1 / naturalWidth) * 100;
        const top = (obj.box.y1 / naturalHeight) * 100;
        const width = ((obj.box.x2 - obj.box.x1) / naturalWidth) * 100;
        const height = ((obj.box.y2 - obj.box.y1) / naturalHeight) * 100;
        return (
          <div
            key={`${obj.label}-${idx}`}
            className="absolute pointer-events-none"
            style={{
              left: `${left}%`,
              top: `${top}%`,
              width: `${width}%`,
              height: `${height}%`,
              border: `2px solid ${color}`,
              boxShadow: `0 0 0 1px rgba(0,0,0,0.25)`,
            }}
          >
            <span
              className="absolute -top-[1px] left-0 -translate-y-full whitespace-nowrap rounded-t px-1 text-[10px] font-medium text-white"
              style={{ backgroundColor: color }}
            >
              {obj.label} {Math.round(obj.confidence * 100)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
