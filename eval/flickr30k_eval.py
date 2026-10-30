"""Đánh giá mô hình train trên Flickr30k bằng đúng quy trình đã dùng cho COCO: chặn logits khi suy luận + chọn lại câu.

Quy trình mỗi ảnh (giống retrieval_logit_bias_experiment.py / sobj_threshold_eval.py, chỉ đổi dữ liệu sang Flickr30k):
  1. YOLO-World (yolov8l-worldv2, ngưỡng 0,25) -> vật thể thấy được.
  2. Truy hồi K = 5 câu gần ảnh nhất trong kho câu của tập TRAIN Flickr30k (CLIP ViT-B/32, như COCO).
  3. Bằng chứng E(I) = YOLO  U  {vật thể được >= r trong K câu truy hồi nhắc tới} (r = 3).
  4. Luật chặn cố định: z' = z - delta * 1[w là từ vật thể COCO, o(w) không thuộc E(I)] tại mỗi bước giải mã (delta = 0 là không chặn).
  5. Sinh 5 ứng viên (1 beam thường + diverse beam search), chấm CLIP ViT-B/16 và FormulaReranker (0,2 / 0,4 / 0,4).
  6. Chọn lại câu: ba tín hiệu (S_base, S_align, S_belief) và năm tín hiệu (+ 1,0 x S_obj + 0,1 x S_retr, min-max theo ảnh).

Hai giai đoạn, để chia ảnh cho tối đa 2 tiến trình song song:
  run    sinh câu ứng viên cho một đoạn ảnh với một danh sách delta, lưu ra "mảnh" JSON (có thể --resume).
  merge  gộp các mảnh, tính BLEU/METEOR/ROUGE-L/CIDEr(/SPICE) và CHAIRs/CHAIRi cho từng delta và từng cách chọn lại câu;
         với --select, chọn delta theo quy tắc của bài (CHAIRs thấp nhất trong các delta mà CIDEr không thấp hơn delta = 0 quá 1 điểm).

Khác COCO: Flickr30k không có COCO instances nên CHAIR dùng nhãn đúng rút từ 5 caption tham chiếu của ảnh (con số cao hơn, KHÔNG so
trực tiếp với CHAIR trên COCO). Kho truy hồi là caption của tập train Flickr30k, không phải COCO.

Ví dụ (từ demo/, Python có torch + transformers + ultralytics; Java cho METEOR):
    # 1) chọn delta trên tập val (1.000 ảnh đầu, 2 mảnh chạy song song)
    python research/scripts/flickr30k_eval.py run --split val --start 0   --max-images 500 --deltas 0 1 2 3 4 \\
        --checkpoint flickr30k_..._best_model.pth --vocab-file flickr30k_..._vocab.json --output results/f30k_val_part0.json
    python research/scripts/flickr30k_eval.py run --split val --start 500 --max-images 500 --deltas 0 1 2 3 4 ... --output results/f30k_val_part500.json
    python research/scripts/flickr30k_eval.py merge --parts results/f30k_val_part0.json results/f30k_val_part500.json \\
        --select --output results/f30k_val_eval.json
    # 2) chạy test với delta đã chọn (ví dụ 3) và delta = 0 để so
    python research/scripts/flickr30k_eval.py run --split test --deltas 0 3 ... --output results/f30k_test_part0.json
    python research/scripts/flickr30k_eval.py merge --parts results/f30k_test_part*.json --output results/f30k_test_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEMO = HERE.parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(HERE))

DATASET = DEMO.parent / "dataset"
FLICKR = DATASET / "flickr30k"
SPLIT_SIZES = (29000, 1014, 1000)   # train / val / test của Karpathy; dùng khi chỉ có file caption (chia ngẫu nhiên, giống notebook)
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ dữ liệu Flickr30k
def load_meta(karpathy_json: Path | None, captions_file: Path | None) -> list[dict]:
    """[{filename, split, sents}]: dataset_flickr30k.json (Karpathy) hoặc captions.txt/results.csv chia ngẫu nhiên như notebook."""
    meta: list[dict] = []
    if karpathy_json and karpathy_json.is_file():
        data = json.loads(karpathy_json.read_text(encoding="utf-8"))
        for img in data["images"]:
            sents = [s["raw"] if "raw" in s else " ".join(s.get("tokens", [])) for s in img.get("sentences", [])]
            if img.get("filename") and sents:
                meta.append({"filename": img["filename"], "split": img.get("split"), "sents": sents})
        log(f"split Karpathy từ {karpathy_json}")
        return meta
    if captions_file and captions_file.is_file():
        by_image: dict[str, list[str]] = {}
        with open(captions_file, "r", encoding="utf-8") as f:
            header = f.readline()
            sep = "|" if "|" in header else ","
            for line in f:
                parts = line.rstrip("\n").split(sep, 2 if sep == "|" else 1)
                if len(parts) >= 2:
                    by_image.setdefault(parts[0].strip(), []).append(parts[-1].strip())
        names = sorted(by_image)
        random.Random(SEED).shuffle(names)
        n_tr, n_va, n_te = SPLIT_SIZES
        for i, name in enumerate(names):
            split = "train" if i < n_tr else "val" if i < n_tr + n_va else "test" if i < n_tr + n_va + n_te else None
            if split:
                meta.append({"filename": name, "split": split, "sents": by_image[name]})
        log(f"CHIA NGẪU NHIÊN từ {captions_file} (không phải split Karpathy; phải trùng cách chia lúc train)")
        return meta
    raise SystemExit("Không thấy --karpathy-json (dataset_flickr30k.json) hay --captions-file (captions.txt/results.csv).")


def retrieval_bank(clip, meta: list[dict], cache_dir: Path) -> tuple[list[str], np.ndarray]:
    """Kho truy hồi: mọi caption của tập train Flickr30k (viết thường), mã hóa bằng CLIP ViT-B/32, lưu cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    caps_f, emb_f = cache_dir / "train_captions.json", cache_dir / "train_caption_emb_b32.npy"
    if caps_f.is_file() and emb_f.is_file():
        return json.loads(caps_f.read_text(encoding="utf-8")), np.load(emb_f)
    caps = [s.strip().lower() for m in meta if m["split"] == "train" for s in m["sents"]]
    log(f"mã hóa kho truy hồi: {len(caps)} caption train")
    emb = clip.text(caps)
    caps_f.write_text(json.dumps(caps), encoding="utf-8")
    np.save(emb_f, emb)
    return caps, emb


# ------------------------------------------------------------------ chặn logits khi giải mã
CURRENT_BIAS: torch.Tensor | None = None


def install_logit_bias() -> None:
    import local_captioner

    orig = local_captioner.ImageCaptioningTransformer.decode_step

    def biased(self, memory, sequence):
        logits = orig(self, memory, sequence)
        return logits if CURRENT_BIAS is None else logits + CURRENT_BIAS

    local_captioner.ImageCaptioningTransformer.decode_step = biased


IRREGULAR = {"men": "man", "women": "woman", "kids": "kid", "guys": "guy", "mice": "mouse"}


def vocab_object_map(vocab: dict[str, int], objects: set[str], inv: dict[str, str]) -> dict[int, str]:
    """id token -> lớp COCO chuẩn, cho các từ vật thể một token trong từ điển (như retrieval_logit_bias_experiment.py)."""
    import chair_metric

    lem = chair_metric._LEMMATIZER
    out = {}
    for word, idx in vocab.items():
        w = IRREGULAR.get(word) or lem.lemmatize(word, pos="n")
        if w in objects:
            out[idx] = inv[w]
    return out


# ------------------------------------------------------------------ giai đoạn run
def cmd_run(args) -> None:
    global CURRENT_BIAS
    if args.device == "cpu":
        torch.cuda.is_available = lambda: False   # các module của pipeline tự chọn thiết bị bằng hàm này
    import chair_metric
    import vocab as vocab_mod
    from clipscore import ClipScorer
    from formula_reranker import FormulaReranker
    from generate import CandidateCaptionGenerator
    from schema import DetectedObject
    from sretr_experiment import Clip
    from yolo_objects import YoloObjectExtractor

    meta = load_meta(args.karpathy_json, args.captions_file)
    split_meta = [m for m in meta if m["split"] == args.split][args.start: args.start + args.max_images]
    if not split_meta:
        raise SystemExit(f"Split '{args.split}' không có ảnh nào trong khoảng đã chọn.")
    image_dir = args.image_dir
    paths = [image_dir / m["filename"] for m in split_meta]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} ảnh trong {image_dir}, ví dụ {missing[:3]}")

    objects_set, inv = chair_metric.load_synonyms()
    clip32 = Clip()
    caps, bank_emb = retrieval_bank(clip32, meta, args.cache_dir)

    # --- bằng chứng (không phụ thuộc delta): YOLO-World + K câu truy hồi
    extractor = YoloObjectExtractor(str(args.yolo_model), args.conf, args.max_objects)
    vocab_mod.set_object_noun_vocab(extractor.class_names())
    log(f"YOLO-World trên {len(paths)} ảnh")
    yolo_objs = [extractor.extract(p) for p in paths]
    img_emb = torch.from_numpy(clip32.images(paths)).to(clip32.device)
    bank_t = torch.from_numpy(bank_emb).to(clip32.device)
    nb = np.concatenate([(img_emb[i: i + 256] @ bank_t.T).topk(args.k, dim=1).indices.cpu().numpy()
                         for i in range(0, len(img_emb), 256)])
    del bank_t

    evidence = []
    for objs, idx in zip(yolo_objs, nb):
        yolo = {inv[o.label] for o in objs if o.confidence >= args.conf and o.label in inv}
        counts = Counter(o for j in idx for o in set(chair_metric.caption_to_words(caps[j], objects_set, inv)[1]))
        maj = {o for o, n in counts.items() if n >= args.maj}
        evidence.append({"yolo": sorted(yolo), "counts": dict(counts), "support": yolo | maj})

    # --- mô hình sinh câu, CLIP ViT-B/16 và bộ chọn lại câu
    vocab = json.loads(args.vocab_file.read_text(encoding="utf-8"))
    tok2obj = vocab_object_map(vocab, objects_set, inv)
    log(f"{len(tok2obj)} token trong từ điển ứng với vật thể COCO")
    install_logit_bias()
    generator = CandidateCaptionGenerator(
        checkpoint_path=args.checkpoint, vocab_path=args.vocab_file, max_candidates=5, num_beams=10,
        num_beam_groups=5, diversity_penalty=0.8, max_new_tokens=40, safe_beam_size=3)
    clip16 = ClipScorer(model_name="openai/clip-vit-base-patch16")
    alpha, beta, gamma = args.weights
    formula = FormulaReranker(alpha=alpha, beta=beta, gamma=gamma, formula_mode="linear", clip_scorer=clip16)

    out = {"split": args.split, "start": args.start, "images": len(split_meta), "maj": args.maj, "k": args.k,
           "checkpoint": str(args.checkpoint), "weights": list(args.weights), "runs": {}}
    if args.resume and args.output.is_file():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if old.get("images") == out["images"] and old.get("start") == out["start"]:
            out["runs"] = old["runs"]
            log("resume: " + ", ".join(f"delta {d}: {len(r['rows'])} ảnh" for d, r in out["runs"].items()))

    for delta in args.deltas:
        key = f"{delta:g}"
        run = out["runs"].setdefault(key, {"delta": delta, "rows": []})
        t0, done0 = time.time(), len(run["rows"])
        for i in range(done0, len(split_meta)):
            m, path, objs, ev = split_meta[i], paths[i], yolo_objs[i], evidence[i]
            if delta:
                bias = torch.zeros(len(vocab))
                for tid, obj in tok2obj.items():
                    if obj not in ev["support"]:
                        bias[tid] = -delta
                CURRENT_BIAS = bias.to(generator.device)
            else:
                CURRENT_BIAS = None
            cands = generator.generate(path, objs)
            scored = clip16.score(path, cands)
            ranked = formula.rerank(scored, objs)
            rs = {r.caption.text: r.rerank_score for r in ranked}
            texts = [c.text for c in scored]
            if texts:
                ce = clip32.text(texts).astype(np.float32)
                s_retr = (ce @ bank_emb[nb[i]].astype(np.float32).T).mean(axis=1)
            else:
                s_retr = np.zeros(0)
            run["rows"].append({
                "image": m["filename"], "references": m["sents"], "first": texts[0] if texts else "",
                "yolo": ev["yolo"], "counts": ev["counts"],
                "objects": [{"label": o.label, "confidence": o.confidence} for o in objs],
                "candidates": [{"text": t, "rerank_score": rs.get(t, -1e9), "s_retr": float(sr),
                                "named": chair_metric.caption_to_words(t, objects_set, inv)[1]}
                               for t, sr in zip(texts, s_retr)]})
            n = i + 1 - done0
            if n % args.save_every == 0:
                args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
                log(f"  delta={key}: {i + 1}/{len(split_meta)} ({(time.time() - t0) / n:.2f}s/ảnh)")
        CURRENT_BIAS = None
        args.output.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        log(f"xong delta={key} ({len(run['rows'])} ảnh)")
    log(f"wrote {args.output}")


# ------------------------------------------------------------------ giai đoạn merge
def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.full_like(v, 0.5) if hi - lo < 1e-9 else (v - lo) / (hi - lo)


def pick(rows: list[dict], mode: str, thr: int = 3, lam_obj: float = 1.0, lam_retr: float = 0.1) -> list[str]:
    """mode 'first' = câu beam thường, 'three' = ba tín hiệu, 'five' = năm tín hiệu (S_obj ngưỡng thr/K, S_retr)."""
    out = []
    for r in rows:
        cands = r["candidates"]
        if not cands:
            out.append(r["first"])
            continue
        base = np.array([c["rerank_score"] for c in cands], dtype=np.float64)
        if mode == "first":
            out.append(r["first"])
            continue
        score = base
        if mode == "five":
            sup = set(r["yolo"]) | {o for o, n in r["counts"].items() if n >= thr}
            s_obj = np.array([-sum(o not in sup for o in c["named"]) for c in cands], dtype=np.float64)
            s_retr = np.array([c["s_retr"] for c in cands], dtype=np.float64)
            score = base + lam_retr * minmax(s_retr) + lam_obj * minmax(s_obj)
        out.append(cands[int(np.argmax(score))]["text"])
    return out


def cmd_merge(args) -> None:
    import chair_metric
    from evaluate_reranked_quality import compute_metrics

    runs: dict[str, dict] = {}
    info = {}
    for f in args.parts:
        part = json.loads(f.read_text(encoding="utf-8"))
        info = {k: part[k] for k in ("split", "maj", "k", "checkpoint", "weights")}
        for key, run in part["runs"].items():
            runs.setdefault(key, {"delta": run["delta"], "rows": {}})["rows"].update({r["image"]: r for r in run["rows"]})
    objects_set, inv = chair_metric.load_synonyms()

    def score(rows, preds, spice):
        refs = [r["references"] for r in rows]
        gt = {r["image"]: chair_metric.ground_truth_from_captions(r["references"], objects_set, inv) for r in rows}
        m = compute_metrics(preds, refs, use_meteor=not args.no_meteor, use_cider=True, use_spice=spice, tokenizer="simple")
        ch = chair_metric.compute_chair({r["image"]: p for r, p in zip(rows, preds)}, gt, objects_set, inv)
        res = {k: round(float(v) * (100 if k == "CIDEr" else 1), 2) for k, v in m.items()}   # như sobj_threshold_eval.py
        res.update(CHAIRs=round(ch["CHAIRs"], 2), CHAIRi=round(ch["CHAIRi"], 2),
                   mean_len=round(float(np.mean([len(p.split()) for p in preds])), 2))
        log(str(res))
        return res

    configs = [("before_rerank", "first", 0)] + [("three_score", "three", 0)] + [(f"five_score_{t}of{info['k']}", "five", t) for t in args.thresholds]
    result = {**info, "gt": "5 caption tham chiếu (Flickr30k không có COCO instances)", "deltas": {}}
    order = None
    for key in sorted(runs, key=lambda x: float(x)):
        rows_by_img = runs[key]["rows"]
        order = order or list(rows_by_img)
        rows = [rows_by_img[i] for i in order if i in rows_by_img]
        log(f"delta = {key}: {len(rows)} ảnh")
        entry = {"images": len(rows)}
        for name, mode, thr in configs:
            entry[name] = score(rows, pick(rows, mode, thr, args.lam_obj, args.lam_retr), args.spice and name == args.spice_on)
        result["deltas"][key] = entry

    if args.select:
        sel_key = args.select_on
        base_key = "0"
        if base_key not in result["deltas"]:
            raise SystemExit("--select cần có delta = 0 trong các mảnh để làm mốc CIDEr.")
        ref_cider = result["deltas"][base_key][sel_key]["CIDEr"]
        ok = {k: v for k, v in result["deltas"].items() if float(k) > 0 and v[sel_key]["CIDEr"] >= ref_cider - args.max_cider_drop}
        result["selection"] = {"rule": f"CHAIRs thấp nhất sau '{sel_key}' trong các delta > 0 có CIDEr >= {ref_cider} - {args.max_cider_drop}",
                               "eligible": sorted(ok, key=float),
                               "chosen": min(ok, key=lambda k: (ok[k][sel_key]["CHAIRs"], -ok[k][sel_key]["CIDEr"])) if ok else None}
        log(f"chọn delta: {result['selection']}")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {args.output}")
    print_table(result)


def print_table(result: dict) -> None:
    cols = ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE", "CHAIRs", "CHAIRi"]
    print("\n" + f"{'delta':>6} {'cách chọn câu':<18}" + "".join(f"{c:>9}" for c in cols))
    for key, entry in result["deltas"].items():
        for name, vals in entry.items():
            if isinstance(vals, dict):
                print(f"{key:>6} {name:<18}" + "".join(f"{vals.get(c, '-'):>9}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="sinh ứng viên cho một đoạn ảnh với các delta")
    r.add_argument("--split", choices=["val", "test"], required=True)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--max-images", type=int, default=100000, help="số ảnh của mảnh này (chia mảnh cho 2 tiến trình)")
    r.add_argument("--deltas", type=float, nargs="+", default=[0.0, 3.0], help="mức chặn delta; 0 = không chặn")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--vocab-file", type=Path, required=True)
    r.add_argument("--karpathy-json", type=Path, default=FLICKR / "dataset_flickr30k.json")
    r.add_argument("--captions-file", type=Path, default=None, help="dự phòng nếu không có Karpathy json (phải khớp cách chia lúc train)")
    r.add_argument("--image-dir", type=Path, default=FLICKR / "images")
    r.add_argument("--cache-dir", type=Path, default=DEMO / "results" / "flickr30k_cache")
    r.add_argument("--yolo-model", type=Path, default=DEMO / "yolov8l-worldv2.pt")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--max-objects", type=int, default=20)
    r.add_argument("--k", type=int, default=5, help="số câu truy hồi")
    r.add_argument("--maj", type=int, default=3, help="ngưỡng đa số r dùng cho bằng chứng khi chặn logits")
    r.add_argument("--weights", type=float, nargs=3, default=(0.2, 0.4, 0.4), metavar=("ALPHA", "BETA", "GAMMA"))
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--save-every", type=int, default=25)
    r.add_argument("--output", type=Path, required=True)
    r.set_defaults(fn=cmd_run)

    m = sub.add_parser("merge", help="gộp các mảnh và tính chỉ số")
    m.add_argument("--parts", type=Path, nargs="+", required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--thresholds", type=int, nargs="+", default=[3, 1], help="ngưỡng r của S_obj cho bộ năm tín hiệu")
    m.add_argument("--lam-obj", type=float, default=1.0)
    m.add_argument("--lam-retr", type=float, default=0.1)
    m.add_argument("--no-meteor", action="store_true", help="bỏ METEOR (nếu không có Java)")
    m.add_argument("--spice", action="store_true", help="tính thêm SPICE (Java 8, chậm), chỉ cho cách chọn câu --spice-on")
    m.add_argument("--spice-on", default="five_score_3of5")
    m.add_argument("--select", action="store_true", help="chọn delta theo quy tắc của bài (chạy trên tập val)")
    m.add_argument("--select-on", default="three_score", help="cách chọn câu dùng để chọn delta (bài dùng bộ ba tín hiệu)")
    m.add_argument("--max-cider-drop", type=float, default=1.0)
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
