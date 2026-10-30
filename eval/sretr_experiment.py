"""Idea D2: retrieval-consensus signal S_retr added to the existing formula reranking.

For each image, retrieve the top-K Karpathy train/restval captions by CLIP image->text
similarity (no val/test captions in the corpus). S_retr(c) = mean cosine between candidate c and
those K captions in CLIP text space. A second variant, S_obj, penalises COCO objects a candidate
names that neither YOLO (conf >= 0.25) nor any of the K neighbours mention.

new_score(c) = rerank_score(c) + lam * minmax_in_image(S)

Equal-weight variants rebuild the linear formula from the per-candidate terms stored in the
report's `reason` string (already min-max normalised per image by FormulaReranker):
    eq3      = (S_base + S_align + S_belief) / 3
    eq4_retr = (S_base + S_align + S_belief + minmax(S_retr)) / 4
    eq4_obj  = (S_base + S_align + S_belief + minmax(S_obj)) / 4

Exploratory: lam and K are swept directly on the test report (no val tuning yet), to see whether
the signal moves CHAIR/CIDEr at all before a proper weight search.

Usage (from demo/, a Python with torch + transformers):
    python research/scripts/sretr_experiment.py \
        --val-report results/reranked_quality_report_dynamic_logits_only_val.json \
        --test-report results/reranked_quality_report_dynamic_logits_only_test.json \
        --extra-test-report results/eval_fixloss_dynamiclogits_rerank_test_belief_penalty.json \
        --output results/sretr_experiment.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO_DIR))

import chair_metric  # noqa: E402
from evaluate_reranked_quality import compute_metrics  # noqa: E402

DATASET_DIR = DEMO_DIR.parent / "dataset"
CACHE_DIR = DATASET_DIR / "retrieval_cache"
MODEL_NAME = "openai/clip-vit-base-patch32"
LAMBDAS = [0.1, 0.2, 0.3, 0.5, 1.0]
KS = [5]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Clip:
    def __init__(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def text(self, texts: list[str], bs: int = 512) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            inp = self.proc(text=texts[i : i + bs], return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                f = self.model.get_text_features(**inp)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(out)

    @torch.no_grad()
    def images(self, paths: list[Path], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i : i + bs]]
            inp = self.proc(images=ims, return_tensors="pt").to(self.device)
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                f = self.model.get_image_features(**inp)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(out)


def corpus(clip: Clip) -> tuple[list[str], np.ndarray]:
    CACHE_DIR.mkdir(exist_ok=True)
    caps_f, emb_f = CACHE_DIR / "train_captions.json", CACHE_DIR / "train_caption_emb_b32.npy"
    if caps_f.is_file() and emb_f.is_file():
        return json.loads(caps_f.read_text(encoding="utf-8")), np.load(emb_f)
    data = json.loads((DATASET_DIR / "dataset_coco.json").read_text(encoding="utf-8"))
    caps = [s["raw"].strip().lower() for im in data["images"] if im["split"] in ("train", "restval") for s in im["sentences"]]
    log(f"encode {len(caps)} train/restval captions")
    emb = clip.text(caps)
    caps_f.write_text(json.dumps(caps), encoding="utf-8")
    np.save(emb_f, emb)
    return caps, emb


def neighbours(clip: Clip, rows: list[dict], corpus_emb: np.ndarray, tag: str, kmax: int) -> np.ndarray:
    f = CACHE_DIR / f"neighbours_{tag}_k{kmax}.npy"
    if f.is_file():
        return np.load(f)
    paths = [DATASET_DIR / "val2014" / "val2014" / r["image"] for r in rows]
    log(f"encode {len(paths)} images ({tag})")
    img = torch.from_numpy(clip.images(paths)).to(clip.device)
    cor = torch.from_numpy(corpus_emb).to(clip.device)
    idx = []
    for i in range(0, len(img), 256):
        sims = img[i : i + 256] @ cor.T
        idx.append(sims.topk(kmax, dim=1).indices.cpu().numpy())
    idx = np.concatenate(idx)
    np.save(f, idx)
    return idx


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.full_like(v, 0.5) if hi - lo < 1e-9 else (v - lo) / (hi - lo)


def scores_for_report(clip, rows, nb_idx, caps, corpus_emb, objects, inv):
    """Per image: candidate texts, stored rerank scores, S_retr[K], S_obj[K]."""
    texts = [c["text"] for r in rows for c in r["candidates"]]
    uniq = sorted(set(texts))
    emb = dict(zip(uniq, clip.text(uniq)))
    out = []
    for r, idx in zip(rows, nb_idx):
        cands = [c["text"] for c in r["candidates"]]
        rs = {x["caption"]: x["rerank_score"] for x in r["ranking"]}
        base = np.array([rs.get(t, -1e9) for t in cands], dtype=np.float64)
        reasons = {x["caption"]: x.get("reason", "") for x in r["ranking"]}
        terms = np.array(
            [[float(m.group(1)) if (m := re.search(rf"{n}=([0-9.]+)", reasons.get(t, ""))) else 0.5
              for n in ("S_base", "S_align", "S_belief")] for t in cands],
            dtype=np.float64,
        )
        ce = np.stack([emb[t] for t in cands]).astype(np.float32)
        ne = corpus_emb[idx].astype(np.float32)
        yolo = {inv[o["label"]] for o in r.get("objects", []) if o["confidence"] >= 0.25 and o["label"] in inv}
        nb_objs = [set(chair_metric.caption_to_words(caps[j], objects, inv)[1]) for j in idx]
        s_retr, s_obj, s_objy = {}, {}, {}
        for k in KS:
            s_retr[k] = (ce @ ne[:k].T).mean(axis=1)
            support = set().union(*nb_objs[:k]) | yolo
            s_obj[k] = np.array(
                [-sum(o not in support for o in chair_metric.caption_to_words(t, objects, inv)[1]) for t in cands],
                dtype=np.float64,
            )
            # Ablation: same penalty with YOLO as the only evidence (no retrieved captions).
            s_objy[k] = np.array(
                [-sum(o not in yolo for o in chair_metric.caption_to_words(t, objects, inv)[1]) for t in cands],
                dtype=np.float64,
            )
        out.append({"cands": cands, "base": base, "terms": terms, "s_retr": s_retr, "s_obj": s_obj, "s_objy": s_objy})
    return out


def select(per_image, signal: str | None, k: int, lam: float) -> list[str]:
    picks = []
    for p in per_image:
        if signal == "eq3":
            score = p["terms"].mean(axis=1)
        elif signal in ("eq4_retr", "eq4_obj"):
            extra = minmax(p["s_retr" if signal == "eq4_retr" else "s_obj"][k])
            score = (p["terms"].sum(axis=1) + extra) / 4
        else:
            score = p["base"].copy()
        if signal == "combo":
            lam_retr, lam_obj = lam
            score = score + lam_retr * minmax(p["s_retr"][k]) + lam_obj * minmax(p["s_obj"][k])
        if signal in ("s_retr", "s_obj", "s_objy") and lam:
            score = score + lam * minmax(p[signal][k])
        picks.append(p["cands"][int(np.argmax(score))])
    return picks


def evaluate(rows, picks, gt_by_fn, objects, inv) -> dict:
    refs = [r["references"] for r in rows]
    m = compute_metrics(picks, refs, use_meteor=False, use_cider=True, use_spice=False, tokenizer="simple")
    ch = chair_metric.compute_chair({r["image"]: p for r, p in zip(rows, picks)}, gt_by_fn, objects, inv)
    return {
        "BLEU-4": round(m["BLEU-4"], 2),
        "CIDEr": round(100 * m["CIDEr"], 2),
        "CHAIRs": round(ch["CHAIRs"], 2),
        "CHAIRi": round(ch["CHAIRi"], 2),
        "mean_len": round(float(np.mean([len(p.split()) for p in picks])), 2),
    }


def chair_gt(rows, objects, inv):
    fn_to_id = chair_metric.load_filename_to_cocoid(DATASET_DIR / "dataset_coco.json")
    inst = chair_metric.load_instance_objects_by_cocoid(DATASET_DIR / "instances_val2014.json", inv)
    return {
        r["image"]: chair_metric.ground_truth_from_captions(r["references"], objects, inv)
        | inst.get(fn_to_id.get(r["image"]), set())
        for r in rows
    }


def sweep(clip, report: Path, tag: str, caps, corpus_emb, objects, inv, configs=None) -> dict:
    rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
    nb = neighbours(clip, rows, corpus_emb, tag, max(KS))
    per_image = scores_for_report(clip, rows, nb, caps, corpus_emb, objects, inv)
    gt = chair_gt(rows, objects, inv)
    results = {"report": str(report), "runs": []}
    configs = configs or (
        [("eq4_retr", 5, None), ("eq4_obj", 5, None)]
        + [(s, k, lam) for s in ("s_retr", "s_obj") for k in KS for lam in LAMBDAS]
    )
    base = evaluate(rows, select(per_image, None, 0, 0.0), gt, objects, inv)
    results["runs"].append({"signal": "none", "K": None, "lambda": 0.0, **base})
    log(f"{tag} formula only: {base}")
    for s, k, lam in configs:
        m = evaluate(rows, select(per_image, s, k, lam), gt, objects, inv)
        results["runs"].append({"signal": s, "K": k, "lambda": lam, **m})
        log(f"{tag} {s} K={k} lam={lam}: {m}")
    return results


PROTOCOL_GROUPS = {
    "s_obj": [("s_obj", 5, lam) for lam in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)],
    "s_objy": [("s_objy", 5, lam) for lam in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)],
    "combo": [("combo", 5, (lr, lo)) for lr in (0.1, 0.2, 0.3) for lo in (0.5, 1.0, 2.0)],
}
MAX_CIDER_DROP = 0.5


def choose_on_val(val: dict) -> dict:
    """Per group: lowest CHAIRs among configs whose CIDEr >= formula-only CIDEr - MAX_CIDER_DROP."""
    base = val["runs"][0]
    chosen = {}
    for group in PROTOCOL_GROUPS:
        ok = [r for r in val["runs"] if r["signal"] == group and r["CIDEr"] >= base["CIDEr"] - MAX_CIDER_DROP]
        if ok:
            chosen[group] = min(ok, key=lambda r: (r["CHAIRs"], -r["CIDEr"]))
    return chosen


def protocol_val(clip, caps, corpus_emb, objects, inv, val_report: Path) -> dict:
    """Step 1: sweep every pre-registered config on val and lock one config per group."""
    configs = [c for group in PROTOCOL_GROUPS.values() for c in group]
    val = sweep(clip, val_report, val_report.stem, caps, corpus_emb, objects, inv, configs)
    chosen = choose_on_val(val)
    log(f"chosen on val: {chosen}")
    return {"criterion": f"min CHAIRs with CIDEr >= base - {MAX_CIDER_DROP}", "val": val, "chosen_on_val": chosen}


def protocol_test(clip, caps, corpus_emb, objects, inv, locked: dict, test_report: Path) -> dict:
    """Step 2 (run once): evaluate only the configs locked on val."""
    test_configs = [
        (r["signal"], r["K"], tuple(r["lambda"]) if isinstance(r["lambda"], list) else r["lambda"])
        for r in locked["chosen_on_val"].values()
    ]
    locked["test"] = sweep(clip, test_report, test_report.stem, caps, corpus_emb, objects, inv, test_configs)
    return locked


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, nargs="+")
    p.add_argument("--protocol-val", type=Path, metavar="VAL_REPORT",
                   help="step 1: sweep pre-registered configs on VAL_REPORT and lock one per group into --output")
    p.add_argument("--protocol-test", type=Path, nargs=2, metavar=("LOCKED_JSON", "TEST_REPORT"),
                   help="step 2: evaluate the configs locked in LOCKED_JSON once on TEST_REPORT")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--combo", action="store_true", help="only run formula + 0.3*S_retr + lam*S_obj")
    args = p.parse_args()

    objects, inv = chair_metric.load_synonyms()
    clip = Clip()
    caps, corpus_emb = corpus(clip)
    if args.protocol_val or args.protocol_test:
        if args.protocol_val:
            result = protocol_val(clip, caps, corpus_emb, objects, inv, args.protocol_val)
        else:
            locked = json.loads(args.protocol_test[0].read_text(encoding="utf-8"))
            if "test" in locked:
                raise SystemExit("test already evaluated for this locked config; refusing to rerun")
            result = protocol_test(clip, caps, corpus_emb, objects, inv, locked, args.protocol_test[1])
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log(f"wrote {args.output}")
        return
    configs = [("combo", 5, (0.3, lo)) for lo in (0.3, 0.5, 1.0, 2.0)] if args.combo else None
    runs = [sweep(clip, r, r.stem, caps, corpus_emb, objects, inv, configs) for r in args.reports]
    args.output.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
