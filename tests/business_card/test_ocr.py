from pathlib import Path

import pytest

from app.common.ocr import run_ocr


def test_business_card_ocr_runs_on_available_sample():
    samples = [
        path for path in Path("data/business_card/test").glob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]
    if not samples:
        pytest.skip("Add a business card image under data/business_card/test/ to run OCR")
    text = run_ocr(str(samples[0]))
    assert isinstance(text, str)
    assert text.strip()
