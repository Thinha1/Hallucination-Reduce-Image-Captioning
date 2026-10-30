from __future__ import annotations

from collections import Counter
from pathlib import Path

from .schema import DetectedObject


class YoloObjectExtractor:
    """Extract object labels and boxes with Ultralytics YOLO."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        max_objects: int = 20,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_objects = max_objects
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency 'ultralytics'. Install it with: pip install ultralytics"
                ) from exc
            self._model = YOLO(self.model_name)
        return self._model

    def class_names(self) -> set[str]:
        """The exact label set this checkpoint recognizes, read from the .pt file itself (not
        hardcoded) -- used to build the object-noun vocabulary for S_belief/hallucination
        flagging (see vocab.py) so it always matches whatever YOLO checkpoint is configured,
        with no separate class list to download or keep in sync."""
        model = self._load_model()
        return {str(name).lower() for name in model.names.values()}

    def extract(self, image_path: str | Path) -> list[DetectedObject]:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        model = self._load_model()
        results = model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        objects: list[DetectedObject] = []
        for result in results:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                objects.append(
                    DetectedObject(
                        label=str(names.get(cls_id, cls_id)),
                        confidence=confidence,
                        box_xyxy=xyxy,
                    )
                )

        objects.sort(key=lambda item: item.confidence, reverse=True)
        return objects[: self.max_objects]


def summarize_objects(objects: list[DetectedObject]) -> str:
    if not objects:
        return "no detected objects"

    counts = Counter(obj.label for obj in objects)
    parts = []
    for label, count in counts.most_common():
        if count == 1:
            parts.append(label)
        else:
            parts.append(f"{count} {label}s")
    return ", ".join(parts)
