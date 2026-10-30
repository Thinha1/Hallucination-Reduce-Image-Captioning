"""Final test evaluation for direction 1 (logit suppression with retrieval evidence).

Compares the unbiased captioner and the gamma-biased captioner at three points:
  before rerank          first candidate (plain beam search)
  old formula rerank     0.2*S_base + 0.4*S_align + 0.4*S_belief
  new formula rerank     old formula + 1*S_obj + 0.1*S_retr (locked on val, section 2)

Usage (from demo/, Python with torch + transformers, Java for METEOR):
    python research/scripts/logit_bias_final_eval.py \
        --parts results/retrieval_logit_bias_test_part0.json results/retrieval_logit_bias_test_part2500.json \
        --output results/logit_bias_final_test.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chair_metric  # noqa: E402
import sretr_experiment as se  # noqa: E402
from evaluate_reranked_quality import compute_metrics  # noqa: E402

DEMO = Path(__file__).resolve().parents[2]
TEST_REPORT = DEMO / "results" / "new_dynamic_cons_attnpool_rerank_test.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-report", type=Path, default=TEST_REPORT, help="unbiased --run-pipeline report (one per seed)")
    ap.add_argument("--spice", action="store_true", help="tính thêm SPICE (Java 8, không dùng cache LMDB ~98GB)")
    ap.add_argument("--spice-sides", default="biased",
                    help="bên nào tính SPICE khi có --spice: 'biased', 'base' hoặc 'base,biased' (mỗi điểm ~30k câu)")
    ap.add_argument("--merged", type=Path, default=DEMO / "results" / "logit_bias_gamma1_test_report.json",
                    help="where to save the biased run as a report")
    args = ap.parse_args()

    base_report = args.base_report.resolve()
    base_data = json.loads(base_report.read_text(encoding="utf-8"))
    # Mốc có thể là report --run-pipeline ("rows") hoặc một lượt retrieval_logit_bias_experiment --save-rows ("runs"),
    # vd model evgate chạy với lớp chặn tắt (--configs 0,0).
    base_rows = base_data["rows"] if "rows" in base_data else \
        [r for r in base_data["runs"] if r["config"] != "report"][0]["rows"]
    biased, config = {}, None
    for f in args.parts:
        run = [r for r in json.loads(f.read_text(encoding="utf-8"))["runs"] if r["config"] != "report"][0]
        config = run["config"]
        for row in run["rows"]:
            biased[row["image"]] = row
    biased_rows = [biased[r["image"]] for r in base_rows]  # same order as the test report

    # Save the biased run as a normal report and reuse the test report's neighbour cache
    # (neighbours depend only on the image, and the row order is identical).
    merged = args.merged
    merged.write_text(json.dumps({"split": "test", "config": config, "rows": biased_rows}, ensure_ascii=False), encoding="utf-8")
    src_nb = se.CACHE_DIR / f"neighbours_{base_report.stem}_k5.npy"
    dst_nb = se.CACHE_DIR / f"neighbours_{merged.stem}_k5.npy"
    if not dst_nb.is_file():
        shutil.copyfile(src_nb, dst_nb)

    objects, inv = chair_metric.load_synonyms()
    clip = se.Clip()
    caps, emb = se.corpus(clip)
    gt = se.chair_gt(base_rows, objects, inv)
    refs = [r["references"] for r in base_rows]

    def new_rerank(rows, stem):
        nb = se.neighbours(clip, rows, emb, stem, max(se.KS))
        per_image = se.scores_for_report(clip, rows, nb, caps, emb, objects, inv)
        return se.select(per_image, "combo", 5, (0.1, 1.0))

    spice_sides = {s.strip() for s in args.spice_sides.split(",")} if args.spice else set()

    def score(captions, side):
        m = compute_metrics(captions, refs, use_meteor=True, use_cider=True, use_spice=side in spice_sides,
                            tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: c for r, c in zip(base_rows, captions)}, gt, objects, inv)
        out = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()}
        out.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2),
                   mean_len=round(float(np.mean([len(c.split()) for c in captions])), 2))
        se.log(str(out))
        return out

    result = {"config": config, "images": len(base_rows), "base": {}, "biased": {}}
    for name, rows, stem in (("base", base_rows, base_report.stem), ("biased", biased_rows, merged.stem)):
        result[name]["before_rerank"] = score([r["baseline_candidate"] for r in rows], name)
        result[name]["old_rerank"] = score([r["caption"] for r in rows], name)
        result[name]["new_rerank"] = score(new_rerank(rows, stem), name)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    se.log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
