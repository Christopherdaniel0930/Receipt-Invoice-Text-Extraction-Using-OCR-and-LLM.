import logging
import tempfile
from pathlib import Path

from app.receipt.schemas.schema import ReceiptData

logger = logging.getLogger(__name__)


def extract_receipt_image(image_bytes: bytes, image_format: str) -> ReceiptData:
    """Pass an uploaded image through the existing receipt processing pipeline."""
    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
            image_file.write(image_bytes)
            temp_path = image_file.name

        logger.info("Starting receipt extraction")
        # Import at call time because app.main also registers this API router.
        from app.main import process

        result = process(temp_path)
        logger.info("Receipt extraction completed")
        return result
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
