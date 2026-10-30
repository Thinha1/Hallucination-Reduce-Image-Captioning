from __future__ import annotations

import re

# Fallback object-noun vocabulary (roughly the COCO-80 classes, hand-typed from memory -- it
# turned out to be missing 11 real COCO classes and to spell some others differently, e.g.
# "racket" vs the real "tennis racket"). Used only until set_object_noun_vocab() is called with
# the *actual* class names of the YOLO checkpoint in use (see YoloObjectExtractor.class_names()),
# which is the source of truth in every real run. This stays as a safety net for code paths that
# never load a YOLO model at all (e.g. unit tests).
COMMON_OBJECT_NOUNS = {
    "person", "man", "woman", "boy", "girl", "child", "kid", "people",
    "bicycle", "car", "motorcycle", "airplane", "plane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "ski", "snowboard",
    "kite", "skateboard", "surfboard", "racket", "bottle", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake",
    "chair", "couch", "sofa", "plant", "bed", "table", "toilet", "tv", "television", "laptop",
    "mouse", "remote", "keyboard", "phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "fridge", "book", "clock", "vase", "scissors", "teddy", "hairdryer",
    "toothbrush",
}

# Populated by set_object_noun_vocab() from the real YOLO checkpoint's class names. None means
# "not set yet" -- fall back to the static COMMON_OBJECT_NOUNS above.
_dynamic_object_nouns: set[str] | None = None


def set_object_noun_vocab(class_names: set[str]) -> None:
    """Replace the object-noun vocabulary with the head noun of each given class name -- meant
    to be called once at startup with YoloObjectExtractor.class_names(), so S_belief and
    hallucination flagging always match whatever YOLO checkpoint is actually configured instead
    of a hand-typed guess.

    Multi-word COCO labels (e.g. "cell phone", "fire hydrant") contribute only their last word
    ("phone", "hydrant") since is_object_noun() matches single caption words -- the last word is
    normally the head noun of an English compound. Labels already plural in COCO's own naming
    (e.g. "skis") get both forms added ("skis" and "ski") so a caption using either still matches.
    """
    global _dynamic_object_nouns
    nouns: set[str] = set()
    for name in class_names:
        words = name.lower().split()
        if not words:
            continue
        head = words[-1]
        nouns.add(head)
        if head.endswith("s") and len(head) > 1:
            nouns.add(head[:-1])
    _dynamic_object_nouns = nouns


def active_vocab() -> set[str]:
    """The vocabulary currently in effect -- the real YOLO class names if set_object_noun_vocab()
    has run, otherwise the static fallback."""
    return _dynamic_object_nouns if _dynamic_object_nouns is not None else COMMON_OBJECT_NOUNS


def clean_word(word: str) -> str:
    """Lowercase and strip anything that isn't a letter (punctuation attached to a word)."""
    return re.sub(r"[^a-zA-Z]", "", word).lower()


def singularize(word: str) -> str:
    """Best-effort singular form, only for words we recognize as plural object nouns
    (e.g. "dogs" -> "dog"); leaves everything else untouched."""
    vocab = active_vocab()
    if word.endswith("s") and word[:-1] in vocab:
        return word[:-1]
    return word


def is_object_noun(word: str) -> bool:
    """Whether `word` (raw, possibly with case/punctuation) names a common visual object."""
    return singularize(clean_word(word)) in active_vocab()
