"""Rerank lại các câu ứng viên đã lưu với vật thể từ một report khác (vd YOLO-World), không sinh câu lại.

Chỉ đúng cho cấu hình KHÔNG chặn logits (YOLO không ảnh hưởng lúc sinh câu). Đọc một file part của
retrieval_logit_bias_experiment.py --save-rows, thay "objects" của mỗi ảnh bằng vật thể trong --objects-report,
chạy lại bộ chọn lại câu công thức (S_base, S_align, S_belief; S_belief cần độ tương đồng CLIP giữa câu và
nhãn vật thể mới, tính bằng ClipScorer ViT-B/16 như lúc sinh) rồi ghi part mới cùng định dạng, để
logit_bias_final_eval.py tính rerank cũ, rerank mới và các chỉ số.

Chạy: python research/scripts/evgate/rerank_with_new_objects.py --part results/A_part.json \
          --objects-report results/X_yoloworld.json --output results/A_yoloworld_part.json
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

import vocab as vocab_mod  # noqa: E402
from clipscore import ClipScorer  # noqa: E402
from llm import CaptionReranker  # noqa: E402
from retrieval_logit_bias_experiment import CACHE  # noqa: E402
from schema import CaptionCandidate, DetectedObject  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=Path, required=True)
    ap.add_argument("--objects-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--neighbours-from", type=Path, default=DEMO / "results" / "seed43_restval_e10_rerank_test.json",
                    help="report có cache láng giềng cùng thứ tự ảnh")
    args = ap.parse_args()

    part = json.loads(args.part.read_text(encoding="utf-8"))
    run = [r for r in part["runs"] if r["config"] != "report"][0]
    obj_data = json.loads(args.objects_report.read_text(encoding="utf-8"))
    new_objects = {r["image"]: r["objects"] for r in obj_data["rows"]}
    vocab_mod.set_object_noun_vocab(set(obj_data["yolo_classes"]))

    reranker = CaptionReranker(provider="formula", clip_scorer=ClipScorer(model_name="openai/clip-vit-base-patch16"))
    t0, changed = time.time(), 0
    for i, row in enumerate(run["rows"]):
        row["objects"] = new_objects[row["image"]]
        objs = [DetectedObject(o["label"], o["confidence"], tuple(o["box_xyxy"]) if o.get("box_xyxy") else None)
                for o in row["objects"]]
        cands = [CaptionCandidate(text=c["text"], source=c.get("source", ""), generation_score=c.get("generation_score"),
                                  clip_score=c.get("clip_score"), metadata=c.get("metadata") or {})
                 for c in row["candidates"]]
        ranked = reranker.rerank(cands, objs)
        best = ranked[0].caption.text if ranked else ""
        changed += best != row["caption"]
        row["caption"] = best
        row["candidates"] = [c.__dict__ for c in cands]  # giữ vc_sim mới tính để lần sau khỏi tính lại
        row["ranking"] = [{"caption": it.caption.text, "rank": it.rank, "rerank_score": it.rerank_score,
                           "reason": it.reason} for it in ranked]
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(run['rows'])} ({(time.time() - t0) / (i + 1):.3f}s/img)", flush=True)

    run["config"] = f"{run['config']}+objects:{args.objects_report.stem}"
    run.pop("predictions", None)
    out = {"runs": [run], "source_part": str(args.part), "objects_report": str(args.objects_report)}
    args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    src = CACHE / f"neighbours_{args.neighbours_from.stem}_k5.npy"
    dst = CACHE / f"neighbours_{args.output.stem}_k5.npy"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    print(f"wrote {args.output}: {changed}/{len(run['rows'])} ảnh đổi câu chọn")


if __name__ == "__main__":
    main()
