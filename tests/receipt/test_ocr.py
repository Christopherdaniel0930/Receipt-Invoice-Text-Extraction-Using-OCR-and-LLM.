from app.common.ocr import RapidOCREngine


def test_ocr_engine_has_recognize():
    assert callable(RapidOCREngine.recognize)
