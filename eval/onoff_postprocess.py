"""Ghép 4 đoạn 1.250 ảnh của một cấu hình bật/tắt thành một báo cáo 5.000 ảnh rồi chấm bộ chọn lại câu năm tín hiệu.

Chạy từ demo/ (mặc định chạy trên CPU để không chiếm GPU của các luồng sinh câu):
    python research/scripts/onoff_postprocess.py --name cs_g3 --evidence union --spice
    python research/scripts/onoff_postprocess.py --name e40_existing --merged results/e40_g3_yoloworld_merged.json
Kết quả: results/onoff/<name>_merged.json (nếu ghép) và results/onoff/<name>_eval.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[2]
CACHE = DEMO.parent / "dataset" / "retrieval_cache"
# tất cả báo cáo cùng 5.000 ảnh kiểm tra theo cùng thứ tự nên dùng chung chỉ số láng giềng truy hồi
NB_SOURCE = CACHE / "neighbours_e40_g3_yoloworld_merged_k5.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--evidence", choices=["union", "yolo", "retr"], default="union")
    ap.add_argument("--merged", type=Path, help="dùng báo cáo đã ghép sẵn thay vì ghép 4 đoạn")
    ap.add_argument("--spice", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="cho phép dùng GPU (mặc định CPU)")
    a = ap.parse_args()
    out_dir = DEMO / "results" / "onoff"
    if a.merged:
        report = a.merged if a.merged.is_absolute() else DEMO / a.merged
    else:
        rows = []
        for start in (0, 1250, 2500, 3750):
            d = json.loads((out_dir / f"{a.name}_part{start}.json").read_text(encoding="utf-8"))
            run = [r for r in d["runs"] if r["config"] != "report"][0]
            rows += run["rows"]
        assert len(rows) == 5000, len(rows)
        report = out_dir / f"{a.name}_merged.json"
        report.write_text(json.dumps({"split": "test", "config": run["config"], "rows": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"đã ghép {len(rows)} ảnh -> {report}")
    nb = CACHE / f"neighbours_{report.stem}_k5.npy"
    if not nb.is_file():
        shutil.copyfile(NB_SOURCE, nb)
    env = dict(os.environ)
    if not a.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "sobj_threshold_eval.py"), "--report", str(report),
           "--output", str(out_dir / f"{a.name}_eval.json"), "--thresholds", "3", "--evidence", a.evidence]
    if a.spice:
        cmd.append("--spice")
    subprocess.run(cmd, check=True, cwd=DEMO, env=env)


if __name__ == "__main__":
    main()
