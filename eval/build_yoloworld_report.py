"""Tạo bản sao của một report rerank, thay vật thể YOLOv8n bằng YOLO-World (yolov8l-worldv2).

Giữ nguyên ảnh, câu tham chiếu, câu mốc; chỉ thay trường "objects" và "yolo_classes", nên mọi script
đánh giá đọc report (retrieval_logit_bias_experiment.py, logit_bias_final_eval.py) dùng được ngay với
bằng chứng YOLO-World. YOLO chạy từng ảnh, ngưỡng 0,25, tối đa 20 vật thể, như lúc tạo bằng chứng train.

Chạy: python research/scripts/evgate/build_yoloworld_report.py --report results/X.json --output results/X_yoloworld.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO / "research" / "scripts"))

from retrieval_logit_bias_experiment import CACHE, IMAGE_DIR  # noqa: E402
from yolo_objects import YoloObjectExtractor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--yolo-model", default="yolov8l-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-objects", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=None, help="chỉ lấy N ảnh đầu (vd tập con 1.000 ảnh val)")
    args = ap.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    if args.max_rows:
        data["rows"] = data["rows"][: args.max_rows]
    extractor = YoloObjectExtractor(args.yolo_model, args.conf, args.max_objects)
    data["yolo_classes"] = sorted(extractor.class_names())
    data["yolo_model"] = args.yolo_model
    t0 = time.time()
    for i, row in enumerate(data["rows"]):
        row["objects"] = [o.__dict__ | {"box_xyxy": list(o.box_xyxy) if o.box_xyxy else None}
                          for o in extractor.extract(IMAGE_DIR / row["image"])]
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(data['rows'])} ({(time.time() - t0) / (i + 1):.2f}s/img)", flush=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # Láng giềng truy hồi không phụ thuộc YOLO: dùng lại cache của report gốc.
    src = CACHE / f"neighbours_{args.report.stem}_k5.npy"
    dst = CACHE / f"neighbours_{args.output.stem}_k5.npy"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    print(f"wrote {args.output} ({len(data['rows'])} rows, {len(data['yolo_classes'])} classes)")


if __name__ == "__main__":
    main()
