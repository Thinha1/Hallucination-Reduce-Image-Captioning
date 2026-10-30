from __future__ import annotations

"""CHAIR (Caption Hallucination Assessment with Image Relevance), from Rohrbach, Hendricks,
Burns, Darrell & Saenko, "Object Hallucination in Image Captioning" (EMNLP 2018) --
https://github.com/LisaAnne/Hallucination. Ported from the paper's Python 2 reference
implementation (utils/chair.py) to Python 3: `pattern.en.singularize` is replaced with
`nltk.stem.WordNetLemmatizer(pos="n")` (same intent -- plural noun to singular form -- both
already verified against this project's nltk install), `nltk.word_tokenize` is unchanged.
data/synonyms.txt is copied verbatim into chair_synonyms.txt next to this file.

Why this exists alongside tune_formula_weights.hallucinated_words(): that proxy reuses the
SAME noun-matching logic FormulaReranker.S_belief is built on, so a weight combo leaning on
S_belief is somewhat guaranteed to score well on it -- not independent evidence. CHAIR's ground
truth instead comes from real, human-authored data never touched by this project's own models:
- MSCOCO instance segmentation annotations (instances_val2014.json, official COCO download --
  see load_instance_objects_by_cocoid), and/or
- The human reference captions already bundled in dataset_coco.json (ground_truth_from_captions).
Neither depends on this project's YOLO detector or object-noun vocabulary, so a hallucination
rate computed this way is safe to use as an independent check on FORMULA_ALPHA/BETA/GAMMA.
"""

import json
from pathlib import Path

import nltk

_LEMMATIZER = nltk.stem.WordNetLemmatizer()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYNONYMS_FILE = SCRIPT_DIR / "chair_synonyms.txt"

# Hardcoded MSCOCO-specific rules, copied verbatim from the original chair.py -------------------
_COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
    "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "home plate", "train track",
]
# Qualifiers like "baby"/"adult" (animal) or "passenger" (vehicle) would otherwise falsely fire
# on the bare MSCOCO object ("baby bird" -> "bird", not "bird" + spurious "baby").
_ANIMAL_WORDS = [
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "animal", "cub",
]
_VEHICLE_WORDS = ["jet", "train"]


def _build_double_word_dict() -> dict[str, str]:
    mapping = {word: word for word in _COCO_DOUBLE_WORDS}
    for word in _ANIMAL_WORDS:
        mapping[f"baby {word}"] = word
        mapping[f"adult {word}"] = word
    for word in _VEHICLE_WORDS:
        mapping[f"passenger {word}"] = word
    mapping["bow tie"] = "tie"
    mapping["toilet seat"] = "toilet"
    mapping["wine glas"] = "wine glass"  # typo fix, kept from the original
    return mapping


_DOUBLE_WORD_DICT = _build_double_word_dict()


def load_synonyms(path: str | Path = DEFAULT_SYNONYMS_FILE) -> tuple[set[str], dict[str, str]]:
    """One MSCOCO category per line, comma-separated synonyms, first entry is the canonical
    ("node") word for that category. Returns (flat set of every synonym word, word -> canonical
    node-word map)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    mscoco_objects: set[str] = set()
    inverse_synonym_dict: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        synonyms = [s.strip() for s in line.split(",")]
        mscoco_objects.update(synonyms)
        for synonym in synonyms:
            inverse_synonym_dict[synonym] = synonyms[0]
    return mscoco_objects, inverse_synonym_dict


# Cache keyed by caption text alone -- safe because mscoco_objects/inverse_synonym_dict are
# loaded once per process (load_synonyms() at startup) and never change afterwards. Matters a
# lot for tune_formula_weights.py: a grid search re-evaluates the SAME handful of candidate
# captions per image across every (alpha, beta, gamma) combo (only which one "wins"
# changes), so without this cache the same caption gets re-tokenized/re-lemmatized by nltk
# hundreds of times over -- nltk.word_tokenize is comparatively expensive, and this was the
# actual bottleneck at grid-search scale (e.g. 5000 images x 286 combos without the cache).
_CAPTION_WORDS_CACHE: dict[str, tuple[list[str], list[str]]] = {}


def caption_to_words(
    caption: str,
    mscoco_objects: set[str],
    inverse_synonym_dict: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Port of CHAIR.caption_to_words(): tokenize, singularize, merge MSCOCO "double words"
    (e.g. "traffic light") into one token, apply the toilet/seat special case, then keep only
    words naming an MSCOCO object. Returns (raw matched words, canonical node words)."""
    cached = _CAPTION_WORDS_CACHE.get(caption)
    if cached is not None:
        return cached

    words = [w.lower() for w in nltk.word_tokenize(caption)]
    words = [_LEMMATIZER.lemmatize(w, pos="n") for w in words]

    merged: list[str] = []
    i = 0
    while i < len(words):
        two = " ".join(words[i : i + 2])
        if two in _DOUBLE_WORD_DICT:
            merged.append(_DOUBLE_WORD_DICT[two])
            i += 2
        else:
            merged.append(words[i])
            i += 1
    words = merged

    # "the seat of the toilet" would otherwise fire on "chair" via the seat/chair synonym.
    if "toilet" in words and "seat" in words:
        words = [w for w in words if w != "seat"]

    matched = [w for w in words if w in mscoco_objects]
    node_words = [inverse_synonym_dict[w] for w in matched]
    result = (matched, node_words)
    _CAPTION_WORDS_CACHE[caption] = result
    return result


def ground_truth_from_captions(
    reference_captions: list[str],
    mscoco_objects: set[str],
    inverse_synonym_dict: dict[str, str],
) -> set[str]:
    """Objects considered "really in the image" because at least one human annotator mentioned
    them in a reference caption. Weaker than instance segmentation (an object can be visible but
    unmentioned in all 5 references), but needs nothing beyond dataset_coco.json, which this
    project already has."""
    objects: set[str] = set()
    for caption in reference_captions:
        _, node_words = caption_to_words(caption, mscoco_objects, inverse_synonym_dict)
        objects.update(node_words)
    return objects


def load_filename_to_cocoid(karpathy_json_path: str | Path) -> dict[str, int]:
    """filename -> the real MSCOCO numeric image id (dataset_coco.json's "cocoid"), needed to
    join this project's filename-keyed reports against instances_val2014.json's image_id-keyed
    annotations."""
    data = json.loads(Path(karpathy_json_path).read_text(encoding="utf-8"))
    return {
        image["filename"]: image["cocoid"]
        for image in data.get("images", [])
        if image.get("filename") and image.get("cocoid") is not None
    }


def load_instance_objects_by_cocoid(
    instances_json_path: str | Path,
    inverse_synonym_dict: dict[str, str],
) -> dict[int, set[str]]:
    """Ground-truth objects per MSCOCO image id from real instance segmentation annotations
    (instances_val2014.json -- official COCO download, human-drawn boxes) -- mirrors CHAIR's
    get_annotations_from_segments(). Fully independent of any detector this project runs."""
    data = json.loads(Path(instances_json_path).read_text(encoding="utf-8"))
    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    objects_by_image: dict[int, set[str]] = {}
    for annotation in data["annotations"]:
        name = id_to_name.get(annotation["category_id"])
        if name is None:
            continue
        node_word = inverse_synonym_dict.get(name)
        if node_word is None:
            continue
        objects_by_image.setdefault(annotation["image_id"], set()).add(node_word)
    return objects_by_image


def compute_chair(
    predictions: dict[str, str],
    ground_truth_objects: dict[str, set[str]],
    mscoco_objects: set[str],
    inverse_synonym_dict: dict[str, str],
) -> dict[str, float]:
    """predictions and ground_truth_objects share the same key space (e.g. image filename) --
    the caller decides what that key is, it just has to match on both sides.

    CHAIRs (per-sentence): fraction of captions naming at least one MSCOCO object absent from
    that image's ground truth.
    CHAIRi (per-instance): fraction of MSCOCO object *mentions*, across all captions, that are
    hallucinated -- captures multi-hallucination captions CHAIRs alone would undercount.

    Images with no ground-truth entry at all are skipped (hallucination relative to nothing is
    undefined), and the skip count is reported so a silent near-empty comparison isn't mistaken
    for a real 0%/100% result.
    """
    num_caps = 0
    num_hallucinated_caps = 0
    hallucinated_word_count = 0
    coco_word_count = 0
    skipped = 0

    for key, caption in predictions.items():
        gt_objects = ground_truth_objects.get(key)
        if not gt_objects:
            skipped += 1
            continue
        _, node_words = caption_to_words(caption, mscoco_objects, inverse_synonym_dict)
        num_caps += 1
        coco_word_count += len(node_words)
        hallucinated_here = [w for w in node_words if w not in gt_objects]
        hallucinated_word_count += len(hallucinated_here)
        if hallucinated_here:
            num_hallucinated_caps += 1

    return {
        "CHAIRs": 100.0 * num_hallucinated_caps / num_caps if num_caps else 0.0,
        "CHAIRi": 100.0 * hallucinated_word_count / coco_word_count if coco_word_count else 0.0,
        "num_captions": num_caps,
        "skipped_no_ground_truth": skipped,
    }
