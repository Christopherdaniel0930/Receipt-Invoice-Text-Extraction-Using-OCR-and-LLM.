import json
from pathlib import Path

import pytest

from app import main as pipeline
from app.receipt.modules import currency_recovery
from app.receipt.modules.currency_recovery import (
    analyse_text,
    leading_currency,
    needs_pixels,
    recover_currency,
    restore_symbols,
    symbol_to_code,
    _glyph_width,
    _symbol_windows,
)
from app.common.ocr import RapidOCREngine

NO_SYMBOL_TEXT = """FRESH MART
GSTIN 29ABCDE1234F1Z5
UPI PAY
TOTAL 1,234.00
"""


class FakeGpt:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def extract(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.payload)

    def classify_document(self, ocr_text):
        """These pipeline fixtures are known receipt inputs."""
        return {"document_type": "receipt_invoice", "confidence": 1.0}


def _stub_pipeline(monkeypatch, payload, category=("Food", 0.9)):
    fake_gpt = FakeGpt(payload)
    monkeypatch.setattr(pipeline, "OllamaGptClient", lambda: fake_gpt)
    monkeypatch.setattr(pipeline, "predict_category", lambda text: category)
    return fake_gpt


def _write(tmp_path, text):
    path = tmp_path / "receipt.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_symbol_to_code_maps_common_tokens():
    assert symbol_to_code("\u20b9") == "INR"
    assert symbol_to_code("Rs.") == "INR"
    assert symbol_to_code("SGD") == "SGD"
    assert symbol_to_code("not a currency") is None


def test_analyse_text_recovers_a_dropped_symbol():
    report = analyse_text(NO_SYMBOL_TEXT)
    assert report["currency"] == "INR"
    assert report["symbol_in_ocr_text"] is False
    assert report["recovered"] is True


def test_analyse_text_sees_a_symbol_that_survived():
    report = analyse_text("FRESH MART\nTOTAL \u20b91,234.00\n")
    assert report["currency"] == "INR"
    assert report["symbol_in_ocr_text"] is True
    assert report["recovered"] is False


def test_analyse_text_gives_up_without_evidence():
    report = analyse_text("FRESH MART\nTOTAL 1,234.00\n")
    assert report["currency"] is None
    assert report["confidence"] == 0.0


def test_restore_symbols_patches_only_amounts():
    patched, count = restore_symbols(NO_SYMBOL_TEXT, "INR")
    assert count == 1
    assert "TOTAL ₹1,234.00" in patched
    assert "GSTIN 29ABCDE1234F1Z5" in patched


def test_restore_symbols_marks_standalone_total_on_bare_page():
    """A payment screen whose only figure is a bare total still counts."""

    text = (
        "To OCEANA HOTEL AND RESIDENCY\n"
        "190\n"
        "Pay again\n"
        "Completed\n"
    )
    patched, count = restore_symbols(text, "INR", "₹")
    assert count == 1
    assert "₹190" in patched


def test_restore_symbols_standalone_pass_leaves_identifiers_alone():
    """Marking a bare total must not spill onto neighbouring ids."""

    text = (
        "UPI transaction ID\n"
        "566869484790\n"
        "Indian Bank 6521\n"
        "Google transaction ID\n"
        "CICAgOiGzPekfw\n"
        "190\n"
    )
    patched, count = restore_symbols(text, "INR", "₹")
    assert count == 1
    assert "₹190" in patched
    assert "566869484790" in patched
    assert "Indian Bank 6521" in patched
    assert "CICAgOiGzPekfw" in patched


def test_restore_symbols_standalone_pass_off_when_a_decimal_exists():
    """A page that already has a money-shaped number is left alone."""

    text = "Total due\n1,234.00\n\n42\n"
    patched, count = restore_symbols(text, "INR", "₹")
    assert count == 1
    assert "₹1,234.00" in patched
    assert "\n42\n" in patched



def test_restore_symbols_leaves_marked_amounts_alone():
    text = "TOTAL \u20b91,234.00\n"
    patched, count = restore_symbols(text, "INR")
    assert patched == text
    assert count == 0


def test_restore_symbols_skips_dates_and_invoice_numbers():
    text = "Invoice No 1234.00\n12/09/2026 45.00\n"
    patched, count = restore_symbols(text, "INR")
    assert patched == text
    assert count == 0


def test_restore_symbols_counts_every_patched_line():
    text = "A 12.00\nB 34.00\nC 56.00\n"
    patched, count = restore_symbols(text, "INR")
    assert count == 3
    assert patched.count("\u20b9") == 3


def test_restore_symbols_without_a_currency_is_a_no_op():
    assert restore_symbols(NO_SYMBOL_TEXT, None) == (NO_SYMBOL_TEXT, 0)


def test_recover_currency_reuses_supplied_rows(tmp_path, monkeypatch):
    path = tmp_path / "receipt.png"
    path.write_bytes(b"not really a png")

    def explode(*args, **kwargs):
        raise AssertionError("rows should have been reused, not re-OCR'd")

    monkeypatch.setattr("app.receipt.modules.currency_recovery.read_rows", explode)
    monkeypatch.setattr("app.receipt.modules.currency_recovery.read_glyphs", lambda *a, **k: [])

    report = recover_currency(path, ocr_text=NO_SYMBOL_TEXT, rows=[])

    assert report["currency"] == "INR"
    assert report["glyph_hits"] == []


def test_load_input_reads_txt_without_ocr(tmp_path, monkeypatch):
    path = _write(tmp_path, NO_SYMBOL_TEXT)

    def explode(*args, **kwargs):
        raise AssertionError("a .txt input must not be OCR'd")

    monkeypatch.setattr(pipeline, "read_ocr_rows", explode)

    text, rows = pipeline.load_input(path)

    assert rows is None
    assert "FRESH MART" in text


def test_process_fills_currency_the_llm_missed(tmp_path, monkeypatch):
    path = _write(tmp_path, NO_SYMBOL_TEXT)
    _stub_pipeline(monkeypatch, {"vendor_name": "Fresh Mart", "currency": None})

    data = pipeline.process(path)

    assert data.currency == "INR"


def test_process_patches_the_text_before_the_llm_sees_it(tmp_path, monkeypatch):
    path = _write(tmp_path, NO_SYMBOL_TEXT)
    GptFakeGpt = _stub_pipeline(monkeypatch, {"vendor_name": "Fresh Mart", "currency": None})

    pipeline.process(path)

    assert "TOTAL \u20b91,234.00" in GptFakeGpt.prompts[0]


def test_process_leaves_a_currency_the_llm_found(tmp_path, monkeypatch):
    path = _write(tmp_path, NO_SYMBOL_TEXT)
    _stub_pipeline(monkeypatch, {"vendor_name": "Fresh Mart", "currency": "USD"})

    data = pipeline.process(path)

    assert data.currency == "USD"


def test_process_survives_a_broken_recovery_layer(tmp_path, monkeypatch):
    path = _write(tmp_path, NO_SYMBOL_TEXT)
    GptFakeGpt = _stub_pipeline(monkeypatch, {"vendor_name": "Fresh Mart", "currency": None})

    def explode(*args, **kwargs):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(pipeline, "analyse_text", explode)

    data = pipeline.process(path)

    assert data.currency is None
    assert data.vendor_name == "Fresh Mart"
    assert "FRESH MART" in GptFakeGpt.prompts[0]


def test_recover_stage_uses_text_layers_without_an_image():
    report = pipeline.recover_stage("unused.txt", NO_SYMBOL_TEXT, None)

    assert report["currency"] == "INR"
    assert report["glyph_hits"] == []


def test_ocr_engine_exposes_boxes():
    assert callable(RapidOCREngine.recognize_rows)


def test_box4_reduces_a_polygon_to_bounds():
    from app.common.ocr import _box4

    assert _box4([[0, 0], [10, 0], [10, 4], [0, 4]]) == [0.0, 0.0, 10.0, 4.0]
    assert _box4([[0, 0], [10, 0]]) is None
    assert _box4(None) is None


def test_the_pipeline_engine_is_cached_per_process():
    import app.common.ocr

    first = app.common.ocr.get_engine()
    second = app.common.ocr.get_engine()

    assert first is second


def test_the_crop_engine_is_never_the_pipeline_engine():
    # A use_det=False call corrupts whichever engine it runs on, so the
    # glyph layer must keep its own instance or every later full-image
    # pass in the batch comes back empty.
    import app.common.ocr
    import app.receipt.modules.currency_recovery

    assert app.receipt.modules.currency_recovery.get_engine() is not app.common.ocr.get_engine()


def test_crop_recognition_does_not_break_the_pipeline_engine():
    from PIL import Image

    from app.receipt.modules.currency_recovery import _has_ink, _symbol_windows, MONEY_RE

    images = Path(__file__).resolve().parent.parent / "data" / "test"
    target = images / "image6.jpeg"

    if not images.is_dir() or not target.exists():
        pytest.skip("sample images are not available")

    baseline, _ = pipeline.load_input(str(target))
    engine = currency_recovery.get_engine()
    _, rows = pipeline.load_input(str(target))
    image = Image.open(target).convert("RGB")
    calls = 0

    for row in rows:
        if not row.get("box") or not MONEY_RE.search(row["text"]):
            continue

        for _, window in _symbol_windows(row["box"], row["text"], image.size):
            crop = image.crop(window)

            if not _has_ink(crop):
                continue

            for scale in (2, 4, 6):
                engine(
                    crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS),
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                )
                calls += 1

    assert calls, "the fixture should produce at least one glyph crop"

    after, after_rows = pipeline.load_input(str(target))

    assert after_rows, "OCR must still work after a use_det=False call on the crop engine"
    assert len(after) == len(baseline)


def test_needs_pixels_is_off_when_the_text_names_a_currency():
    assert needs_pixels("FRESH MART\nTOTAL \u20b91,234.00\n") is False
    assert needs_pixels("FRESH MART\nTOTAL 1,234.00\n") is True


def test_needs_pixels_is_off_when_locale_merely_outscores_the_symbol():
    # decide() names "locale" here because the trailing marker items move
    # the running total last, but a symbol was read off the page.
    text = "FRESH MART\nTOTAL \u20b91,234.00\nGSTIN 29ABCDE1234F1Z5\nUPI PAY\n"

    assert analyse_text(text)["source"] == "locale"
    assert needs_pixels(text) is False


def test_needs_pixels_is_on_when_only_locale_markers_agree():
    assert needs_pixels(NO_SYMBOL_TEXT) is True


def test_recover_currency_skips_the_pixel_layer_when_unneeded(tmp_path, monkeypatch):
    path = tmp_path / "receipt.png"
    path.write_bytes(b"not really a png")

    def explode(*args, **kwargs):
        raise AssertionError("the image must not be re-read")

    monkeypatch.setattr("app.receipt.modules.currency_recovery.read_glyphs", explode)

    report = recover_currency(
        path,
        ocr_text="FRESH MART\nTOTAL \u20b91,234.00\n",
        rows=[],
    )

    assert report["currency"] == "INR"
    assert "pixel layer skipped" in " ".join(report["notes"])


def test_recover_currency_runs_the_pixel_layer_when_needed(tmp_path, monkeypatch):
    path = tmp_path / "receipt.png"
    path.write_bytes(b"not really a png")

    called = []
    monkeypatch.setattr(
        "app.receipt.modules.currency_recovery.read_glyphs",
        lambda *a, **k: called.append(1) or [],
    )

    recover_currency(path, ocr_text=NO_SYMBOL_TEXT, rows=[])

    assert called == [1]


def test_leading_currency_accepts_a_glued_glyph():
    assert leading_currency("\u20b91") == "\u20b9"
    assert leading_currency("$12.50") == "$"
    assert leading_currency("12.50") is None
    assert leading_currency("") is None


def test_glyph_width_follows_the_text_not_the_line_height():
    box = (100, 300, 160, 500)
    assert _glyph_width(box, "123.45") == 10.0
    assert _glyph_width(box, "1234567") < 10.0


def test_symbol_windows_are_one_glyph_wide():
    box = (100, 300, 160, 320)
    windows = dict(_symbol_windows(box, "123.45", (1000, 1000)))

    assert set(windows) == {"pre", "lead"}
    assert windows["pre"][2] <= box[0] + 1
    assert windows["lead"][0] >= box[0] - 5
