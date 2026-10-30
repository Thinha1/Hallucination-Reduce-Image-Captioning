#!/usr/bin/env bash
# Thí nghiệm bật/tắt thành phần trên tập kiểm tra (5.000 ảnh, chia 4 đoạn 1.250 ảnh, bỏ qua đoạn đã có kết quả).
# Hai luồng chạy song song (tối đa 2 tiến trình GPU):
#   bash Graduation_Thesis/eval/run_onoff_ablation.sh A   # cơ sở + chặn δ=3 (cs_g3), rồi đề xuất chỉ YOLO (e40_yolo)
#   bash Graduation_Thesis/eval/run_onoff_ablation.sh B   # đề xuất không chặn (e40_g0), rồi đề xuất chỉ truy hồi (e40_retr)
# Chạy từ thư mục demo/. Kết quả: results/onoff/<tên>_part<start>.json
set -u
PY="/c/Users/thinh/AppData/Local/Programs/Python/Python310/python.exe"
REPORT=results/seed43_restval_e10_rerank_test_yoloworld.json
VOCAB=../dataset/coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json
E40=restval_dynlogits_contrastive_attnpool_epoch40_model.pth
BASE=restval_baseline_best_model.pth
mkdir -p results/onoff

run() {  # name checkpoint gamma evidence
  local name=$1 ck=$2 gamma=$3 ev=$4
  for start in 0 1250 2500 3750; do
    out=results/onoff/${name}_part${start}.json
    [ -f "$out" ] && { echo "skip $out"; continue; }
    "$PY" Graduation_Thesis/eval/retrieval_logit_bias_experiment.py --report "$REPORT" --checkpoint "$ck" --vocab-file "$VOCAB" \
      --start "$start" --max-images 1250 --maj 3 --evidence "$ev" --configs "0,$gamma" --save-rows \
      --output "$out" > "results/onoff/${name}_part${start}.log" 2>&1
  done
}

case "${1:-}" in
  A) run cs_g3 "$BASE" 3 union; run e40_yolo "$E40" 3 yolo ;;
  B) run e40_g0 "$E40" 0 union; run e40_retr "$E40" 3 retr ;;
  *) echo "usage: $0 A|B"; exit 1 ;;
esac
echo "done $1"
