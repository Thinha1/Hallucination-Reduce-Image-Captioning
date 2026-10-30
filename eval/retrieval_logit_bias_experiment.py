"""Direction 1: retrieval-guided logit adjustment during decoding.

At every decoding step the captioner's logits get an object-level bias built from two evidence
sources -- YOLO detections and the majority of the top-5 retrieved train captions:

    z'(w) = z(w) + beta  * 1[ obj(w) in YOLO u N_maj ]
                 - gamma * 1[ w is a COCO object word and obj(w) not in YOLO u N_maj ]

N_maj = COCO objects named by >= MAJ of the 5 retrieved captions. Non-object words are untouched.

Everything else is the stock pipeline (CandidateCaptionGenerator -> ClipScorer -> formula
reranker), with YOLO objects and retrieval neighbours reused from an existing --run-pipeline
report so the only change is the bias. beta = gamma = 0 must reproduce the report exactly.

Usage (from demo/, Python with torch + transformers):
    python research/scripts/retrieval_logit_bias_experiment.py \
        --report results/proposed_rerank_val.json --max-images 1000 \
        --configs 0,0 0,1 0,2 0,4 1,2 --output results/retrieval_logit_bias_val1000.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEMO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO))

import chair_metric  # noqa: E402
import local_captioner  # noqa: E402
import vocab as vocab_mod  # noqa: E402
from clipscore import ClipScorer  # noqa: E402
from evaluate_reranked_quality import compute_metrics  # noqa: E402
from generate import CandidateCaptionGenerator  # noqa: E402
from llm import CaptionReranker  # noqa: E402
from schema import DetectedObject  # noqa: E402

DATASET = DEMO.parent / "dataset"
CACHE = DATASET / "retrieval_cache"
IMAGE_DIR = DATASET / "val2014" / "val2014"
IRREGULAR = {"men": "man", "women": "woman", "kids": "kid", "guys": "guy", "mice": "mouse"}

# Bias added to the logits of every decode_step; set per image before generation.
CURRENT_BIAS: torch.Tensor | None = None
_orig_decode_step = local_captioner.ImageCaptioningTransformer.decode_step


def _biased_decode_step(self, memory, sequence):
    logits = _orig_decode_step(self, memory, sequence)
    if CURRENT_BIAS is not None:
        logits = logits + CURRENT_BIAS
    return logits


local_captioner.ImageCaptioningTransformer.decode_step = _biased_decode_step


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vocab_object_map(vocab: dict[str, int], objects: set[str], inv: dict[str, str]) -> dict[int, str]:
    """token id -> canonical COCO object, for single-token object words in the caption vocab."""
    lem = chair_metric._LEMMATIZER
    out = {}
    for word, idx in vocab.items():
        w = IRREGULAR.get(word) or lem.lemmatize(word, pos="n")
        if w in objects:
            out[idx] = inv[w]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--neighbours", type=Path)
    ap.add_argument("--checkpoint", default=str(DEMO / "dynlogits_contrastive_attnpool_best_model.pth"))
    ap.add_argument("--vocab-file", default=str(DATASET / "coco2014_karpathy_baseline_vocab.json"))
    ap.add_argument("--start", type=int, default=0, help="first row (for sharding)")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--maj", type=int, default=3, help="neighbours (of 5) that must name an object")
    ap.add_argument("--evidence", choices=["union", "yolo", "retr"], default="union",
                    help="nguồn bằng chứng E(I) của luật chặn: union = YOLO ∪ truy hồi (mặc định), yolo = chỉ YOLO, retr = chỉ truy hồi")
    ap.add_argument("--configs", nargs="+", default=None, help="beta,gamma pairs (mặc định 0,0 0,1 0,2 0,4 1,2)")
    ap.add_argument("--gate-scales", default="1",
                    help="với --gate-checkpoint: các hệ số k nhân vào bảng β đã học, vd '1,2,3' (k = 1 là β gốc)")
    ap.add_argument("--save-rows", action="store_true", help="also save full report-format rows (all candidates) per run")
    ap.add_argument("--gate-checkpoint", type=Path,
                    help="checkpoint evgate: dùng bảng β đã học (evidence_gate) thay cho --configs; chạy đúng 1 cấu hình 'gate'")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    global CURRENT_BIAS
    data = json.loads(args.report.read_text(encoding="utf-8"))
    rows = data["rows"][args.start : args.start + args.max_images]
    vocab_mod.set_object_noun_vocab(set(data["yolo_classes"]))
    nb_file = args.neighbours or CACHE / f"neighbours_{args.report.stem}_k5.npy"
    nb = np.load(nb_file)[args.start : args.start + len(rows), :5]
    caps = json.loads((CACHE / "train_captions.json").read_text(encoding="utf-8"))
    objects_set, inv = chair_metric.load_synonyms()

    generator = CandidateCaptionGenerator(
        checkpoint_path=Path(args.checkpoint), vocab_path=Path(args.vocab_file),
        max_candidates=5, num_beams=10, num_beam_groups=5, diversity_penalty=0.8, max_new_tokens=40, safe_beam_size=3,
    )
    clip = ClipScorer(model_name="openai/clip-vit-base-patch16")
    reranker = CaptionReranker(provider="formula", clip_scorer=clip)
    vocab = json.loads(Path(args.vocab_file).read_text(encoding="utf-8"))
    tok2obj = vocab_object_map(vocab, objects_set, inv)
    obj_tok_ids = torch.tensor(sorted(tok2obj), dtype=torch.long)
    log(f"{len(tok2obj)} vocab tokens map to COCO objects")

    # Lớp chặn đã học trong notebook evgate: bias(ảnh) = β[mã bằng chứng của từng vật thể] @ ma trận token->vật thể.
    # Mã = số câu truy hồi nhắc vật thể + (K + 1) * [YOLO thấy], đúng như EvidenceGate khi train.
    gate = None
    if args.gate_checkpoint:
        ck = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=False)
        sd, info = ck["model_state_dict"], ck["evidence_gate"]
        if info["mode"] == "contextual":
            raise SystemExit("EvidenceGate 'contextual' cần hidden state của decoder, script này chưa hỗ trợ.")
        g_beta = (-torch.nn.functional.softplus(sd["evidence_gate.beta_raw"]) if "evidence_gate.beta_raw" in sd
                  else sd["evidence_gate.beta"]).float()
        g_tm = sd["evidence_gate.token_matrix"].float()
        assert g_tm.shape[1] == len(vocab), f"token_matrix {tuple(g_tm.shape)} không khớp vocab {len(vocab)}"
        gate = {"beta": g_beta, "tm": g_tm, "objects": info["objects"], "k": info["retr_k"], "yolo": info["use_yolo"]}
        # Cấu hình "gate*k": dùng k·β. --configs (nếu có) chạy thêm sau, cùng model.
        args.configs = [f"gate*{k}" for k in args.gate_scales.split(",")] + (args.configs or [])
        log(f"gate {info['mode']}: β = {[round(v, 3) for v in g_beta.tolist()]} | {int(g_tm.sum())} token, YOLO={gate['yolo']}")

    # Per-image evidence: YOLO objects + majority-of-neighbours objects.
    evidence, gate_codes = [], []
    for r, idx in zip(rows, nb):
        yolo = {inv[o["label"]] for o in r["objects"] if o["confidence"] >= 0.25 and o["label"] in inv}
        sets = [set(chair_metric.caption_to_words(caps[j], objects_set, inv)[1]) for j in idx]
        maj = {w for w in set().union(*sets) if sum(w in s for s in sets) >= args.maj}
        evidence.append({"union": yolo | maj, "yolo": yolo, "retr": maj}[args.evidence])
        if gate is not None:
            gate_codes.append(torch.tensor(
                [sum(o in s for s in sets[: gate["k"]]) + (gate["k"] + 1) * int(gate["yolo"] and o in yolo)
                 for o in gate["objects"]], dtype=torch.long))

    fn2id = chair_metric.load_filename_to_cocoid(DATASET / "dataset_coco.json")
    inst = chair_metric.load_instance_objects_by_cocoid(DATASET / "instances_val2014.json", inv)
    gt = {r["image"]: chair_metric.ground_truth_from_captions(r["references"], objects_set, inv) | inst.get(fn2id.get(r["image"]), set()) for r in rows}
    refs = [r["references"] for r in rows]

    def evaluate(first, best):
        out = {}
        for name, preds in (("before_rerank", first), ("after_rerank", best)):
            m = compute_metrics(preds, refs, use_meteor=False, use_cider=True, use_spice=False, tokenizer="simple")
            ch = chair_metric.compute_chair({r["image"]: p for r, p in zip(rows, preds)}, gt, objects_set, inv)
            out[name] = {"BLEU-4": round(m["BLEU-4"], 2), "CIDEr": round(100 * m["CIDEr"], 2), "CHAIRs": round(ch["CHAIRs"], 2),
                         "CHAIRi": round(ch["CHAIRi"], 2), "mean_len": round(float(np.mean([len(p.split()) for p in preds])), 2)}
        return out

    results = {"report": str(args.report), "start": args.start, "images": len(rows), "maj": args.maj, "evidence": args.evidence, "runs": []}
    # Reference: the report's own captions on the same images (no bias, generated earlier).
    results["runs"].append({"config": "report", **evaluate([r["baseline_candidate"] for r in rows], [r["caption"] for r in rows])})
    log(f"report reference: {results['runs'][-1]}")

    if args.configs is None:
        args.configs = ["0,0", "0,1", "0,2", "0,4", "1,2"]
    for cfg in args.configs:
        is_gate = cfg.startswith("gate")
        scale = float(cfg.split("*")[1]) if "*" in cfg else 1.0
        beta, gamma = (None, None) if is_gate else (float(x) for x in cfg.split(","))
        first, best, identical, full_rows = [], [], 0, []
        t0 = time.time()
        for i, (r, ev) in enumerate(zip(rows, evidence)):
            if is_gate:
                CURRENT_BIAS = (scale * gate["beta"][gate_codes[i]] @ gate["tm"]).to(generator.device)
            else:
                bias = torch.zeros(len(vocab))
                for tid, obj in tok2obj.items():
                    bias[tid] = beta if obj in ev else -gamma
                CURRENT_BIAS = bias.to(generator.device) if (beta or gamma) else None
            image = IMAGE_DIR / r["image"]
            objs = [DetectedObject(o["label"], o["confidence"], tuple(o["box_xyxy"]) if o.get("box_xyxy") else None) for o in r["objects"]]
            cands = generator.generate(image, objs)
            scored = clip.score(image, cands)
            ranked = reranker.rerank(scored, objs, image_path=image)
            first.append(scored[0].text if scored else "")
            best.append(ranked[0].caption.text if ranked else "")
            identical += best[-1] == r["caption"]
            if args.save_rows:
                full_rows.append({
                    "image": r["image"], "caption": best[-1], "baseline_candidate": cands[0].text if cands else None,
                    "references": r["references"], "objects": r["objects"],
                    "candidates": [c.__dict__ for c in scored],
                    "ranking": [{"caption": it.caption.text, "rank": it.rank, "rerank_score": it.rerank_score, "reason": it.reason} for it in ranked],
                })
            if (i + 1) % 200 == 0:
                log(f"  beta={beta} gamma={gamma}: {i + 1}/{len(rows)} ({(time.time() - t0) / (i + 1):.2f}s/img)")
        CURRENT_BIAS = None
        run = {"config": (f"gate*{scale:g}" if scale != 1 else "gate") if is_gate else f"beta={beta:g},gamma={gamma:g}", "beta": beta, "gamma": gamma,
               "same_caption_as_report": identical, **evaluate(first, best),
               "predictions": [{"image": r["image"], "first": f, "best": b} for r, f, b in zip(rows, first, best)]}
        if args.save_rows:
            run["rows"] = full_rows
        results["runs"].append(run)
        log(f"{ {k: v for k, v in run.items() if k not in ('predictions', 'rows')} }")
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
