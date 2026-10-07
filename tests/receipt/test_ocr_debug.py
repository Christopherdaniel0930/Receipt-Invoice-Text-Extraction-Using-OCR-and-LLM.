"""Tests for the box extraction and annotated-image saving in app.common.ocr_debug.

The OCR engine is not exercised here. The drawing path is fed synthetic rows so
it can be checked without cv2 or an ONNX runtime.
"""

import pytest
from PIL import Image

from app.common import ocr_debug


def make_box(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class FakeResult:
    def __init__(self, txts, scores, boxes):
        self.txts = txts
        self.scores = scores
        self.boxes = boxes


def make_rows():
    return [
        {"index": 1, "text": "TOTAL", "score": 0.94, "box": [10.0, 20.0],
         "polygon": make_box(10, 20, 100, 20)},
        {"index": 2, "text": "CASH", "score": 0.65, "box": [10.0, 60.0],
         "polygon": make_box(10, 60, 100, 20)},
        {"index": 3, "text": "TAX", "score": 0.42, "box": [10.0, 100.0],
         "polygon": make_box(10, 100, 100, 20)},
        {"index": 4, "text": "no box", "score": 0.5, "box": None, "polygon": None},
    ]


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "receipt.png"
    Image.new("RGB", (400, 300), "white").save(path)
    return path


def test_polygon_keeps_every_corner():
    assert ocr_debug._polygon(make_box(1, 2, 30, 40)) == [
        [1.0, 2.0], [31.0, 2.0], [31.0, 42.0], [1.0, 42.0],
    ]


@pytest.mark.parametrize("box", [None, [], [[1, 2]], "not a box", 5])
def test_polygon_rejects_unusable_boxes(box):
    assert ocr_debug._polygon(box) is None


def test_corners_still_reports_the_top_left_point():
    assert ocr_debug._corners(make_box(1, 2, 30, 40)) == [1.0, 2.0]


def test_extract_carries_polygons_alongside_the_corner():
    result = FakeResult(
        txts=["TOTAL", "CASH"],
        scores=[0.94, 0.42],
        boxes=[make_box(10, 20, 100, 20), make_box(10, 60, 100, 20)],
    )
    rows = ocr_debug._extract(result)
    assert [row["text"] for row in rows] == ["TOTAL", "CASH"]
    assert [row["score"] for row in rows] == [0.94, 0.42]
    assert rows[0]["polygon"] == make_box(10, 20, 100, 20)
    assert rows[0]["box"] == [10.0, 20.0]


def drawable_rows():
    return [row for row in make_rows() if row["polygon"]]


def row_without_box():
    return [row for row in make_rows() if not row["polygon"]][0]


def test_draw_boxes_saves_an_annotated_copy(image_path, tmp_path):
    target = tmp_path / "annotated.png"
    saved, drawn = ocr_debug.draw_boxes(image_path, make_rows(), target)

    assert saved == target
    assert saved.exists()
    assert drawn == len(drawable_rows())

    with Image.open(saved) as annotated:
        assert annotated.size == Image.open(image_path).size
        assert annotated.getcolors(maxcolors=1_000_000) != [(400 * 300, (255, 255, 255))]


def test_draw_boxes_leaves_the_original_untouched(image_path, tmp_path):
    before = image_path.read_bytes()
    ocr_debug.draw_boxes(image_path, make_rows(), tmp_path / "annotated.png")
    assert image_path.read_bytes() == before


def test_draw_boxes_accepts_a_directory_target(image_path, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    saved, _ = ocr_debug.draw_boxes(image_path, make_rows(), out)
    assert saved == out / "receipt.boxes.png"
    assert saved.exists()


def test_draw_boxes_adds_a_png_suffix_when_missing(image_path, tmp_path):
    saved, _ = ocr_debug.draw_boxes(image_path, make_rows(), tmp_path / "annotated")
    assert saved.suffix == ".png"


def test_draw_boxes_creates_missing_parent_directories(image_path, tmp_path):
    saved, _ = ocr_debug.draw_boxes(image_path, make_rows(), tmp_path / "a" / "b" / "out.png")
    assert saved.exists()


def test_draw_boxes_skips_rows_without_a_polygon(image_path, tmp_path):
    _, drawn = ocr_debug.draw_boxes(image_path, [row_without_box()], tmp_path / "out.png")
    assert drawn == 0


def colors_of(path):
    with Image.open(path) as image:
        return {color for _, color in image.convert("RGB").getcolors(maxcolors=1_000_000)}


def test_box_color_thresholds():
    assert ocr_debug._box_color(0.95) == ocr_debug.BOX_OK
    assert ocr_debug._box_color(0.60) == ocr_debug.BOX_MID
    assert ocr_debug._box_color(0.30) == ocr_debug.BOX_LOW
    assert ocr_debug._box_color(None) == ocr_debug.BOX_LOW
    assert ocr_debug._box_color(0.85, min_score=0.90) == ocr_debug.BOX_LOW


def test_min_score_only_pulls_borderline_boxes_down(image_path, tmp_path):
    strict = Image.open(ocr_debug.draw_boxes(image_path, make_rows(), tmp_path / "a.png", min_score=0.8)[0])
    assert strict.convert("RGB").getpixel((60, 20)) == ocr_debug.BOX_OK
    assert strict.convert("RGB").getpixel((60, 60)) == ocr_debug.BOX_LOW
    assert strict.convert("RGB").getpixel((60, 100)) == ocr_debug.BOX_LOW


def test_low_confidence_boxes_are_red_without_min_score(image_path, tmp_path):
    colors = colors_of(ocr_debug.draw_boxes(image_path, make_rows(), tmp_path / "a.png")[0])
    assert ocr_debug.BOX_OK in colors
    assert ocr_debug.BOX_MID in colors
    assert ocr_debug.BOX_LOW in colors


def test_outline_sits_on_the_polygon_and_leaves_the_interior_alone(image_path, tmp_path):
    rows = [{"index": 1, "text": "A", "score": 0.95,
             "box": [100.0, 150.0],
             "polygon": [[100, 150], [300, 150], [300, 200], [100, 200]]}]
    saved, _ = ocr_debug.draw_boxes(image_path, rows, tmp_path / "a.png")
    annotated = Image.open(saved).convert("RGB")
    background = Image.open(image_path).convert("RGB")

    for point in ((100, 150), (300, 150), (300, 200), (100, 200), (200, 150), (100, 175)):
        assert annotated.getpixel(point) == ocr_debug.BOX_OK, point

    for x in range(105, 296, 5):
        for y in range(155, 196, 5):
            assert annotated.getpixel((x, y)) == background.getpixel((x, y)), (x, y)


def test_label_plate_sits_above_the_box_and_leaves_the_top_edge_visible(image_path, tmp_path):
    rows = [{"index": 1, "text": "A", "score": 0.95,
             "box": [100.0, 150.0],
             "polygon": [[100, 150], [300, 150], [300, 200], [100, 200]]}]
    saved, _ = ocr_debug.draw_boxes(image_path, rows, tmp_path / "a.png")
    annotated = Image.open(saved).convert("RGB")

    above = {annotated.getpixel((x, y)) for x in range(102, 160, 3) for y in range(120, 148)}
    assert ocr_debug.LABEL_FILL in above
    assert (255, 255, 255) in above
    assert annotated.getpixel((200, 150)) == ocr_debug.BOX_OK


def test_label_plate_flips_below_the_box_near_the_top_edge(image_path, tmp_path):
    rows = [{"index": 1, "text": "A", "score": 0.95,
             "box": [10.0, 4.0],
             "polygon": [[10, 4], [200, 4], [200, 40], [10, 40]]}]
    saved, _ = ocr_debug.draw_boxes(image_path, rows, tmp_path / "a.png")
    annotated = Image.open(saved).convert("RGB")

    below = {annotated.getpixel((x, y)) for x in range(12, 60, 3) for y in range(42, 60)}
    assert ocr_debug.LABEL_FILL in below
    assert annotated.getpixel((100, 4)) == ocr_debug.BOX_OK


def test_label_text_switches_on_the_numbered_flag():
    row = {"index": 7, "score": 0.91}
    assert ocr_debug._label_text(row, numbered=True) == "#7 0.91"
    assert ocr_debug._label_text(row, numbered=False) == "0.91"


def test_save_image_flag_is_registered():
    args = ocr_debug.build_parser().parse_args(["receipt.jpg", "--save-image", "out.png"])
    assert args.save_image == "out.png"
