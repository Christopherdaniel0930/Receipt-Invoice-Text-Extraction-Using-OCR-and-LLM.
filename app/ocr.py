from pathlib import Path

from rapidocr import RapidOCR


def _as_float(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _box4(box):
    """Reduce an OCR polygon to (x0, y0, x1, y1), or None if unusable."""

    if box is None:
        return None

    try:
        points = [(float(point[0]), float(point[1])) for point in box]
    except (TypeError, IndexError, ValueError):
        return None

    if len(points) < 3:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


_ENGINE = None


def get_engine() -> RapidOCR:
    """Return the pipeline's OCR engine, created once per process.

    A new ONNX session per image exhausts resources on long scans and
    reloads the det/cls/rec models for nothing. The glyph-crop layer in
    app.currency_recovery keeps a separate instance on purpose, because
    a use_det=False call corrupts whichever engine it runs on.
    """

    global _ENGINE

    if _ENGINE is None:
        _ENGINE = RapidOCR()

    return _ENGINE


class RapidOCREngine:
    """OCR adapter using RapidOCR with its PP-OCRv6 ONNX models."""

    def __init__(self):
        self.engine = get_engine()

    def recognize_rows(self, image_path: str) -> list:
        """Return [{text, score, box}, ...] with a 4-number box.

        The box is what later stages need to crop a dropped symbol back
        out of the image, so it is kept rather than thrown away.
        """

        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image.resolve()}")

        result = self.engine(str(image.resolve()))

        # RapidOCR versions expose OCR text through result.txts.
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = getattr(result, "boxes", None)

        if texts is None and isinstance(result, (tuple, list)) and result:
            first = result[0]
            texts = getattr(first, "txts", None)
            scores = getattr(first, "scores", None)
            boxes = getattr(first, "boxes", None)

        if not texts:
            return []

        scores = list(scores) if scores is not None else []
        boxes = list(boxes) if boxes is not None else []

        rows = []

        for index, text in enumerate(texts):
            rows.append({
                "text": str(text).strip(),
                "score": _as_float(scores[index]) if index < len(scores) else None,
                "box": _box4(boxes[index]) if index < len(boxes) else None,
            })

        return rows

    def recognize(self, image_path: str) -> str:
        return "\n".join(
            row["text"] for row in self.recognize_rows(image_path) if row["text"]
        )


def create_ocr() -> RapidOCREngine:
    return RapidOCREngine()


def read_ocr_rows(image_path: str) -> list:
    """OCR an image once and return the rows with their ink positions."""

    return create_ocr().recognize_rows(image_path)


def run_ocr(image_path: str) -> str:
    return create_ocr().recognize(image_path)
