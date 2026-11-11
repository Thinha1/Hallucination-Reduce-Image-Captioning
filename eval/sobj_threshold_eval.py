"""Ngưỡng bằng chứng của S_obj trên tập kiểm tra: vật thể được miễn phạt khi YOLO thấy hoặc ít nhất r trong K = 5
câu truy hồi nhắc tới (r = 1 là cấu hình đang dùng, tái lập số của signal_table.py; r = 3 là biến thể đa số).
Trọng số giữ như cũ: w_obj = 1, w_retr = 0.1.

Chạy (thư mục demo):
    python research/scripts/sobj_threshold_eval.py --report results/e35_k6_yoloworld_merged.json \
        --output results/e35_k6_sobj_threshold_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chair_metric  # noqa: E402
import sretr_experiment as se  # noqa: E402
from evaluate_reranked_quality import compute_metrics  # noqa: E402

K = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--evidence", choices=["union", "yolo", "retr"], default="union",
                    help="nguồn bằng chứng E(I) của S_obj: union = YOLO ∪ truy hồi (mặc định), yolo = chỉ YOLO, retr = chỉ truy hồi")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--final-only", action="store_true",
                    help="chỉ chấm cấu hình năm tín hiệu ở các ngưỡng --thresholds (bỏ qua các cấu hình khác, dùng khi chấm SPICE bổ sung)")
    ap.add_argument("--spice", action="store_true", help="tính thêm SPICE (Java 8, ~2,5 phút mỗi cấu hình)")
    args = ap.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = data["rows"] if "rows" in data else [r for r in data["runs"] if r["config"] != "report"][0]["rows"]
    objects, inv = chair_metric.load_synonyms()
    clip = se.Clip()
    caps, emb = se.corpus(clip)
    gt = se.chair_gt(rows, objects, inv)
    refs = [r["references"] for r in rows]
    nb = se.neighbours(clip, rows, emb, args.report.stem, max(se.KS))
    per_image = se.scores_for_report(clip, rows, nb, caps, emb, objects, inv)

    def objs(t):
        return chair_metric.caption_to_words(t, objects, inv)[1]

    # số câu truy hồi (trong K) nhắc tới mỗi vật thể; YOLO lọc như sretr_experiment (conf >= 0.25)
    counts, yolo = [], []
    for r, idx in zip(rows, nb):
        counts.append(Counter(o for j in idx[:K] for o in set(objs(caps[j]))))
        yolo.append({inv[o["label"]] for o in r.get("objects", []) if o["confidence"] >= 0.25 and o["label"] in inv})

    def s_obj(thr):
        out = []
        for p, c, y in zip(per_image, counts, yolo):
            ret = {o for o, n in c.items() if n >= thr}
            sup = {"union": y | ret, "yolo": y, "retr": ret}[args.evidence]
            out.append(np.array([-sum(o not in sup for o in objs(t)) for t in p["cands"]], dtype=np.float64))
        return out

    ref_objs = [set().union(*[set(objs(c)) for c in r["references"]]) for r in rows]

    def pick(scores):
        return [p["cands"][int(np.argmax(s))] for p, s in zip(per_image, scores)]

    def score(captions):
        m = compute_metrics(captions, refs, use_meteor=True, use_cider=True, use_spice=args.spice, tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: c for r, c in zip(rows, captions)}, gt, objects, inv)
        out = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()}
        out.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2),
                   mean_len=round(float(np.mean([len(c.split()) for c in captions])), 2))
        # độ phủ đối tượng: số vật thể COCO khác nhau mỗi câu, % câu không nhắc vật thể COCO nào,
        # recall theo vật thể mà ít nhất một câu tham chiếu nhắc tới (chỉ tính ảnh có vật thể như vậy)
        mentioned = [set(objs(c)) for c in captions]
        rec = [len(m & ro) / len(ro) for m, ro in zip(mentioned, ref_objs) if ro]
        out.update(obj_per_cap=round(float(np.mean([len(m) for m in mentioned])), 3),
                   no_obj_pct=round(100 * float(np.mean([not m for m in mentioned])), 2),
                   recall_ref=round(100 * float(np.mean(rec)), 2))
        se.log(str(out))
        return out

    res = {}
    if not args.final_only:
        res = {"before_rerank": score([r["baseline_candidate"] for r in rows]),
               "three_score": score(se.select(per_image, None, K, 0.0)),
               "five_score_ref": score(se.select(per_image, "combo", K, (0.1, 1.0)))}
    for thr in args.thresholds:
        so = s_obj(thr)
        if not args.final_only:
            res[f"plus_s_obj_1_{thr}of5"] = score(pick([p["base"] + 1.0 * se.minmax(m) for p, m in zip(per_image, so)]))
        res[f"five_score_{thr}of5"] = score(pick([p["base"] + 0.1 * se.minmax(p["s_retr"][K]) + 1.0 * se.minmax(m)
                                                  for p, m in zip(per_image, so)]))
        res[f"sobj_equal_{thr}of5"] = round(100 * float(np.mean([m.max() == m.min() for m in so])), 1)
    args.output.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    se.log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
