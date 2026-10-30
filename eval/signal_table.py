"""Bảng "hai tín hiệu mới" (ba điểm, +S_retr, +S_obj, cả hai) từ một report/part đã lưu, đủ BLEU-1..4, METEOR,
ROUGE-L, CIDEr, CHAIRs, CHAIRi trên tập kiểm tra.

Chạy: python research/scripts/evgate/signal_table.py --report results/X_part.json --output results/X_signals.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO / "research" / "scripts"))

import chair_metric  # noqa: E402
import sretr_experiment as se  # noqa: E402
from evaluate_reranked_quality import compute_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True, help="report ('rows') hoặc part ('runs' có 'rows')")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--neighbours-from", type=Path, default=DEMO / "results" / "seed43_restval_e10_rerank_test.json")
    ap.add_argument("--retr-weights", type=float, nargs="*", default=[],
                    help="thêm các dòng +λ_retr·S_retr và năm điểm (λ_retr, 1) với các λ_retr này")
    args = ap.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = data["rows"] if "rows" in data else [r for r in data["runs"] if r["config"] != "report"][0]["rows"]
    dst = se.CACHE_DIR / f"neighbours_{args.report.stem}_k5.npy"
    if not dst.is_file():
        shutil.copyfile(se.CACHE_DIR / f"neighbours_{args.neighbours_from.stem}_k5.npy", dst)

    objects, inv = chair_metric.load_synonyms()
    clip = se.Clip()
    caps, emb = se.corpus(clip)
    gt = se.chair_gt(rows, objects, inv)
    refs = [r["references"] for r in rows]
    nb = se.neighbours(clip, rows, emb, args.report.stem, max(se.KS))
    per_image = se.scores_for_report(clip, rows, nb, caps, emb, objects, inv)

    def score(captions):
        m = compute_metrics(captions, refs, use_meteor=True, use_cider=True, use_spice=False, tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: c for r, c in zip(rows, captions)}, gt, objects, inv)
        out = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()}
        out.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2),
                   mean_len=round(float(np.mean([len(c.split()) for c in captions])), 2))
        se.log(str(out))
        return out

    res = {
        "before_rerank": score([r["baseline_candidate"] for r in rows]),
        "three_score": score(se.select(per_image, None, 5, 0.0)),
        "plus_s_retr_0.1": score(se.select(per_image, "s_retr", 5, 0.1)),
        "plus_s_obj_1": score(se.select(per_image, "s_obj", 5, 1.0)),
        "five_score": score(se.select(per_image, "combo", 5, (0.1, 1.0))),
    }
    for w in args.retr_weights:
        res[f"plus_s_retr_{w:g}"] = score(se.select(per_image, "s_retr", 5, w))
        res[f"five_score_retr{w:g}"] = score(se.select(per_image, "combo", 5, (w, 1.0)))
    args.output.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    se.log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
