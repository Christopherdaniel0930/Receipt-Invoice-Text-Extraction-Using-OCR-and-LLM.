"""Standalone OCR inspection tool.

Runs RapidOCR on an image and prints what the OCR engine actually
produced, so OCR quality can be debugged without touching the
extraction pipeline.

This module is self-contained and is NOT part of the extraction
pipeline. Nothing in the pipeline imports it.

Usage:
    python -m app.ocr_debug data/test/1.jpg
    python -m app.ocr_debug receipt.png --numbered --min-score 0.80
    python -m app.ocr_debug receipt.png --save out.txt
    python -m app.ocr_debug receipt.png --classify
    python -m app.ocr_debug receipt.png --save-image annotated.png

Print the text after every pipeline process:
    python -m app.ocr_debug receipt.jpg --trace
    python -m app.ocr_debug receipt.jpg --trace --llm
    python -m app.ocr_debug --stages
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

LINE_RULE = "-" * 72
HEAVY_RULE = "=" * 72

BOX_OK = (0, 158, 0)
BOX_MID = (255, 145, 0)
BOX_LOW = (214, 0, 0)
LABEL_FILL = (0, 0, 0)
SCORE_LOW = 0.5
SCORE_HIGH = 0.8


def _unwrap(result):
    """RapidOCR returns an output object or a legacy (result, elapse) tuple."""

    texts = getattr(result, "txts", None)

    if texts is None and isinstance(result, (tuple, list)) and result:
        texts = getattr(result[0], "txts", None)

    return texts or []


def _listify(value):
    """Coerce a RapidOCR attribute to a list without numpy truth-value errors."""

    if value is None:
        return []

    try:
        return list(value)
    except TypeError:
        return []


def _extract(result):
    """Return a list of {index, text, score, box} dicts from a RapidOCR result."""

    texts = _unwrap(result)
    scores = _listify(getattr(result, "scores", None))
    boxes = _listify(getattr(result, "boxes", None))

    rows = []

    for index, text in enumerate(texts):
        score = scores[index] if index < len(scores) else None
        box = boxes[index] if index < len(boxes) else None

        rows.append(
            {
                "index": index + 1,
                "text": str(text),
                "score": round(float(score), 4) if score is not None else None,
                "box": _corners(box),
                "polygon": _polygon(box),
            }
        )

    return rows


def _corners(box):
    """Reduce an OCR polygon to (x, y) of its top-left point."""

    try:
        return [round(float(box[0][0]), 1), round(float(box[0][1]), 1)]
    except (TypeError, IndexError, ValueError):
        return None


def _polygon(box):
    """Return the full OCR polygon as a list of (x, y) pairs, or None."""

    if box is None:
        return None

    try:
        points = [(float(point[0]), float(point[1])) for point in box]
    except (TypeError, IndexError, ValueError):
        return None

    if len(points) < 3:
        return None

    return [[round(x, 1), round(y, 1)] for x, y in points]


def _elapsed(result):
    value = getattr(result, "elapse", None)

    if isinstance(value, (int, float)):
        return round(float(value), 2)

    if isinstance(value, (list, tuple)) and value:
        return round(sum(float(item) for item in value), 2)

    return None


def _score_label(score):
    if score is None:
        return "  ???? "
    return f"{score:5.2f}"


_ENGINE = None


def get_engine():
    """Return a shared RapidOCR engine, created once per process.

    Building a new ONNX session per image exhausts resources on long
    scans, so the engine is cached.
    """

    global _ENGINE

    if _ENGINE is None:
        from rapidocr import RapidOCR

        _ENGINE = RapidOCR()

    return _ENGINE


def read_image(image_path):
    """Run OCR and return the raw RapidOCR result object."""

    source = Path(image_path)

    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source.resolve()}")

    return get_engine()(str(source.resolve()))


def print_ocr(
    rows,
    normalized,
    numbered=False,
    show_boxes=False,
    min_score=None,
):
    """Print OCR rows followed by the normalized text."""

    print(HEAVY_RULE)
    print("RAW OCR")
    print(HEAVY_RULE)

    if not rows:
        print("(no text recognised)")
        return

    weak = []

    for row in rows:
        marker = ""
        if min_score is not None and row["score"] is not None and row["score"] < min_score:
            marker = "  <-- LOW"
            weak.append(row)

        prefix = f"[{row['index']:>3}] " if numbered else ""
        suffix = ""

        if show_boxes and row["box"]:
            suffix = f"  @ {row['box']}"

        print(f"{prefix}{_score_label(row['score'])}  {row['text']}{suffix}{marker}")

    print(LINE_RULE)
    print(f"lines: {len(rows)}"
          f"   low-confidence: {len(weak)}"
          f"   mean score: {_mean_score(rows)}")
    print(LINE_RULE)

    if weak:
        print("\nlowest scoring lines:")
        for row in sorted(weak, key=lambda item: item["score"] or 0)[:10]:
            print(f"  {_score_label(row['score'])}  {row['text']}")
        print()

    if normalized is None:
        return

    print(HEAVY_RULE)
    print("NORMALIZED TEXT (what the pipeline would send)")
    print(HEAVY_RULE)
    print(normalized)
    print(HEAVY_RULE)


def _mean_score(rows):
    scores = [row["score"] for row in rows if row["score"] is not None]

    if not scores:
        return "n/a"

    return f"{sum(scores) / len(scores):.3f}"


def _box_color(score, min_score=None):
    """Green for confident text, orange for middling, red for suspect."""

    if score is None:
        return BOX_LOW

    if min_score is not None and score < min_score:
        return BOX_LOW

    if score >= SCORE_HIGH:
        return BOX_OK

    if score >= SCORE_LOW:
        return BOX_MID

    return BOX_LOW


def _load_font(size):
    """Load a scalable font, falling back to the built-in bitmap one."""

    for name in ("DejaVuSans.ttf", "Arial.ttf", "segoeui.ttf", "Tahoma.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    return ImageFont.load_default()


def _label_text(row, numbered):
    parts = []

    if numbered:
        parts.append(f"#{row['index']}")

    if row["score"] is not None:
        parts.append(f"{row['score']:.2f}")

    return " ".join(parts)


def _draw_label(canvas, text, x, y, bottom, font):
    """Write text on a filled plate above the box, or below it near the top edge."""

    if not text:
        return

    left, top, right, box_bottom = canvas.textbbox((0, 0), text, font=font)
    width = right - left
    height = box_bottom - top

    plate_top = y - height - 4

    if plate_top < 0:
        plate_top = bottom + 4

    canvas.rectangle(
        [x - 2, plate_top - 2, x + width + 2, plate_top + height + 2],
        fill=LABEL_FILL,
    )
    canvas.text((x - left, plate_top - top), text, fill=(255, 255, 255), font=font)


def _resolve_output(image_path, out_path):
    """Turn a directory, bare name or file path into a concrete image path."""

    target = Path(out_path)
    stem = Path(image_path).stem

    if target.is_dir():
        target = target / f"{stem}.boxes.png"

    if not target.suffix:
        target = target.with_suffix(".png")

    return target


def draw_boxes(image_path, rows, out_path, min_score=None, numbered=True):
    """Draw every OCR polygon on a copy of the image and save it.

    Returns (saved_path, drawn_count). The original image is never modified.
    """

    image = Image.open(image_path).convert("RGB")
    canvas = ImageDraw.Draw(image)
    font = _load_font(max(12, image.height // 60))
    drawn = 0

    for row in rows:
        points = row.get("polygon")

        if not points:
            continue

        color = _box_color(row.get("score"), min_score)
        closed = [(int(round(x)), int(round(y))) for x, y in points]
        closed.append(closed[0])

        canvas.line(closed, fill=color, width=2)
        drawn += 1

        if numbered or row["score"] is not None:
            bottom = max(y for _, y in closed)
            _draw_label(canvas, _label_text(row, numbered), closed[0][0], closed[0][1], bottom, font)

    saved = _resolve_output(image_path, out_path)
    saved.parent.mkdir(parents=True, exist_ok=True)
    image.save(saved)

    return saved, drawn


STAGE_RULE = "#" * 72

STAGES = (
    "01 raw text",
    "02 normalize_ocr_text",
    "03 classifier input",
    "04 build_extraction_prompt",
    "05 llm response",
    "06 json.loads",
    "07 normalize_llm_response",
    "08 model_validate",
    "09 normalize_data",
    "10 deterministic_validate",
    "11 expense category",
)


def print_block(title, body, note=None):
    """Print a titled text block, the unit of the stage trace."""

    print(STAGE_RULE)
    print(f"STAGE: {title}")
    if note:
        print(f"       {note}")
    print(STAGE_RULE)

    if body is None:
        print("(skipped)")
    elif body == "":
        print("(empty)")
    else:
        print(body)

    print()


def print_diff(label, before, after):
    """Print a unified diff between two stage outputs."""

    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()

    if before_lines == after_lines:
        print(f"  {label}: no change")
        print()
        return

    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="previous",
        tofile="after",
        lineterm="",
        n=1,
    )

    print(f"  {label}:")
    for line in diff:
        print(f"    {line}")
    print()


class TextTrace:
    """Runs the pipeline one process at a time, printing the text after each.

    Every method returns the text produced by its stage, so stages can be
    inspected individually without running the whole trace.

    This is a debugging tool. It does not modify or wrap the pipeline.
    """

    def __init__(self, source, call_llm=False, show_diff=True):
        use_utf8_output()

        self.source = source
        self.call_llm = call_llm
        self.show_diff = show_diff
        self.stages = {}
        self.values = {}
        self.data = None

    def _value(self, name):
        """Return the real object produced by a stage, not its display text."""

        return self.values.get(name)

    def stage_source(self):
        """1. Raw text: OCR for images, file read for .txt."""

        path = Path(self.source)

        if path.exists() and path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
            kind = "read from .txt"
        else:
            rows = _extract(read_image(self.source))
            text = "\n".join(row["text"] for row in rows)
            kind = f"OCR: {len(rows)} lines"

        return self._record("01 raw text", text, note=kind)

    def stage_normalize_ocr(self):
        """2. app.text_normalizer.normalize_ocr_text."""

        from app.text_normalizer import normalize_ocr_text

        text = normalize_ocr_text(self._value("01 raw text"))

        return self._record(
            "02 normalize_ocr_text",
            text,
            note="whitespace collapsed, blank and duplicate lines dropped",
        )

    def stage_classifier_input(self):
        """3. Text handed to the document classifier, plus its verdict."""

        from app.doc_classifier import _normalise, classify_document

        cleaned = self._value("02 normalize_ocr_text")
        verdict = classify_document(cleaned)
        self.verdict = verdict

        print_diff("classifier", cleaned, _normalise(cleaned))

        note = (
            f"verdict={verdict['document_type']} "
            f"is_receipt_invoice={verdict['is_receipt_invoice']} "
            f"score={verdict['score']} confidence={verdict['confidence']}"
        )

        print_block("03 classifier verdict", _dump(verdict), note)

        self.values["03 classifier input"] = cleaned
        self.stages["03 classifier input"] = cleaned

        return cleaned

    # def stage_prompt(self):
    #     """4. app.llm_validator.build_extraction_prompt."""

    #     from app.llm_validator import build_extraction_prompt

    #     prompt = build_extraction_prompt(self._value("03 classifier input"))

    #     return self._record(
    #         "04 build_extraction_prompt",
    #         prompt,
    #         note=f"{len(prompt)} chars sent to the model",
    #     )

    def stage_llm_response(self):
        """5. Raw model response. Requires call_llm=True and network access."""

        if not self.call_llm:
            self._record("05 llm response", None, note="skipped (pass --llm to enable)")
            return None

        from app.llm_validator import OllamaGptClient

        try:
            response = OllamaGptClient().extract(self._value("04 build_extraction_prompt"))
        except Exception as error:
            self._record(
                "05 llm response", None, note=f"FAILED: {type(error).__name__}: {error}"
            )
            return None

        return self._record("05 llm response", response, note="raw completion")

    def stage_json_parse(self):
        """6. json.loads on the model response."""

        response = self._value("05 llm response")

        if not response:
            print_block("06 json.loads", None, "skipped, no model response")
            return None

        try:
            data = json.loads(response)
        except json.JSONDecodeError as error:
            print_block("06 json.loads", None, f"FAILED: {error}")
            return None

        return self._record(
            "06 json.loads", data, display=_dump(data), note="parsed dict re-dumped"
        )

    def stage_normalize_llm(self):
        """7. app.extractor.normalize_llm_response."""

        data = self._value("06 json.loads")

        if data is None:
            print_block("07 normalize_llm_response", None, "skipped")
            return None

        from app.extractor import normalize_llm_response

        normalized = normalize_llm_response(data)

        return self._record(
            "07 normalize_llm_response",
            normalized,
            display=_dump(normalized),
            note="line item field-name variants collapsed",
        )

    def stage_model_validate(self):
        """8. ReceiptData.model_validate, the step that raised your error."""

        data = self._value("07 normalize_llm_response")

        if data is None:
            print_block("08 model_validate", None, "skipped")
            return None

        from app.schema import ReceiptData

        try:
            validated = ReceiptData.model_validate(data)
        except ValidationError as error:
            print_block("08 model_validate", None, f"FAILED:\n{error}")
            return None

        self.data = validated

        return self._record(
            "08 model_validate", validated, display=_dump_json(validated), note="pydantic ReceiptData"
        )

    def stage_normalize_data(self):
        """9. app.validator.normalize_data."""

        if self.data is None:
            print_block("09 normalize_data", None, "skipped, nothing validated")
            return None

        from app.validator import normalize_data

        normalized = normalize_data(self.data)

        return self._record(
            "09 normalize_data",
            normalized,
            display=_dump_json(normalized),
            note="dates, currency, amounts",
        )

    def stage_validation_checks(self):
        """10. app.validator.deterministic_validate."""

        if self.data is None:
            print_block("10 deterministic_validate", None, "skipped")
            return None

        from app.validator import deterministic_validate

        errors = deterministic_validate(self.data)

        return self._record(
            "10 deterministic_validate",
            errors,
            display="\n".join(errors) if errors else "(no errors)",
            note=f"{len(errors)} error(s)",
        )

    def stage_category(self):
        """11. build_classifier_text and predict_category."""

        if self.data is None:
            print_block("11 expense category", None, "skipped")
            return None

        from app.category_model import predict_category

        parts = []

        if self.data.vendor_name:
            parts.append(self.data.vendor_name)
        for item in self.data.line_items:
            parts.append(item.description or "")

        text = " ".join(parts).strip()

        try:
            category, confidence = predict_category(text)
        except FileNotFoundError as error:
            return self._record("11 expense category", None, note=f"model missing: {error}")
        except Exception as error:
            return self._record(
                "11 expense category", None, note=f"FAILED: {type(error).__name__}: {error}"
            )

        return self._record(
            "11 expense category",
            category,
            note=f"confidence={confidence:.3f} on {len(text)} chars",
        )

    def _record(self, name, value, display=None, note=None):
        """Store the real value and print its display text."""

        self.values[name] = value
        self.stages[name] = value if display is None else display

        print_block(name, self.stages[name], note)

        return value

    def run(self):
        """Execute every stage in order and return the stage dict."""

        self.stage_source()
        self.stage_normalize_ocr()
        self.stage_classifier_input()
        # self.stage_prompt()
        self.stage_llm_response()
        self.stage_json_parse()
        self.stage_normalize_llm()
        self.stage_model_validate()
        self.stage_normalize_data()
        self.stage_validation_checks()
        self.stage_category()

        print(HEAVY_RULE)
        print(
            f"TRACE COMPLETE: "
            f"{len([v for v in self.values.values() if v is not None])} stage(s) produced text"
        )
        print(HEAVY_RULE)

        return self.stages


def _dump(data):
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _dump_json(model):
    return json.dumps(model.model_dump(), indent=2, ensure_ascii=False)


def print_every_process(source, call_llm=False, show_diff=True):
    """Print the text produced after every pipeline process for one input."""

    use_utf8_output()

    trace = TextTrace(source, call_llm=call_llm, show_diff=show_diff)
    trace.run()
    return trace


def build_parser():
    parser = argparse.ArgumentParser(
        description="Print RapidOCR output for an image (debugging tool)",
    )
    parser.add_argument("image", nargs="*", help="Image file(s) to OCR")
    parser.add_argument(
        "--numbered",
        action="store_true",
        help="Prefix each line with its index",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Show top-left x,y of each detected text box",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Flag lines below this OCR confidence, e.g. 0.80",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print only the raw OCR text, no report",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        default=None,
        help="Write the OCR text to FILE (raw text, no report)",
    )
    parser.add_argument(
        "--save-image",
        metavar="FILE",
        default=None,
        help="Draw the OCR boxes onto a copy of the image and save it to FILE, "
             "or to <image>.boxes.png if FILE is a directory",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Also run the standalone rule-based document classifier",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit rows as JSON instead of a report",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the text after every pipeline process, stage by stage",
    )
    parser.add_argument(
        "--llm",
        dest="call_llm",
        action="store_true",
        help="With --trace, also call the model (needs network and .env)",
    )
    parser.add_argument(
        "--stages",
        action="store_true",
        help="List the trace stages and exit",
    )

    return parser


def print_stage_list():
    """Print the stage order used by the trace."""

    print(STAGE_RULE)
    print("TRACE STAGES")
    print(STAGE_RULE)

    for name in STAGES:
        print(f"  {name}")

    print(STAGE_RULE)
    print(f"  {len(STAGES)} stages. --llm adds the model call.")
    print(STAGE_RULE)


def use_utf8_output():
    """Force UTF-8 on stdout/stderr so currency symbols survive the console."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def run(argv=None):
    """Entry point. Returns a process exit code."""

    use_utf8_output()

    args = build_parser().parse_args(argv)

    if args.stages:
        print_stage_list()
        return 0

    if not args.image:
        print("error: provide at least one image", file=sys.stderr)
        return 2

    if args.trace:
        for source in args.image:
            try:
                print_every_process(source, call_llm=args.call_llm)
            except FileNotFoundError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
        return 0

    if not args.raw and not args.as_json:
        from app.text_normalizer import normalize_ocr_text

    exit_code = 0

    for image in args.image:
        try:
            result = read_image(image)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            exit_code = 1
            continue
        except Exception as error:
            print(f"error: {image}: {type(error).__name__}: {error}", file=sys.stderr)
            exit_code = 1
            continue

        rows = _extract(result)
        text = "\n".join(row["text"] for row in rows)
        normalized = None if (args.raw or args.as_json) else normalize_ocr_text(text)

        if args.save:
            Path(args.save).write_text(text, encoding="utf-8")
            print(f"saved {len(rows)} lines to {args.save}")

        if args.save_image:
            try:
                saved, drawn = draw_boxes(
                    image,
                    rows,
                    args.save_image,
                    min_score=args.min_score,
                    numbered=args.numbered,
                )
            except Exception as error:
                print(
                    f"error: {args.save_image}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                exit_code = 1
            else:
                print(f"saved {drawn} of {len(rows)} box(es) on {saved}")

        if args.as_json:
            print(json.dumps(
                {
                    "image": image,
                    "elapse": _elapsed(result),
                    "rows": rows,
                },
                indent=2,
                ensure_ascii=False,
            ))
            continue

        if args.raw:
            print(text)
            continue

        print_ocr(
            rows,
            normalized,
            numbered=args.numbered,
            show_boxes=args.show_boxes,
            min_score=args.min_score,
        )

        if args.classify:
            from app.doc_classifier import classify_document

            verdict = classify_document(normalized)
            print(HEAVY_RULE)
            print(
                f"CLASSIFIER: {verdict['document_type']}  "
                f"is_receipt_invoice={verdict['is_receipt_invoice']}  "
                f"score={verdict['score']}  "
                f"confidence={verdict['confidence']}"
            )
            print(f"  matched: {', '.join(verdict['reasons']) or '-'}")
            if verdict["counters"]:
                print(f"  against: {', '.join(verdict['counters'])}")
            for reason in verdict.get("blocked_by", []):
                print(f"  blocked: {reason}")
            print(HEAVY_RULE)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
