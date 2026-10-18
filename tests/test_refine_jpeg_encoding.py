"""Refine-tab JPEG encoding adapted from PR #56.

RGBA and palette PNGs cannot be saved as JPEG until alpha is composited.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image

from utils.image_utils import encode_refine_image_as_jpeg


def _center_pixel(jpeg_bytes, xy):
    with Image.open(BytesIO(jpeg_bytes)) as decoded:
        rgb = decoded.convert("RGB")
        return rgb.getpixel(xy)


def _assert_close(actual, expected, tol=2):
    assert all(abs(a - e) <= tol for a, e in zip(actual, expected)), (actual, expected)


def test_rgba_upload_composites_alpha_over_white():
    image = Image.new("RGBA", (32, 16), (255, 0, 0, 0))
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), (0, 0, 255, 255))

    raw = BytesIO()
    try:
        image.save(raw, format="JPEG")
        saved = True
    except OSError:
        saved = False
    assert saved is False

    jpeg_bytes = encode_refine_image_as_jpeg(image)
    assert image.mode == "RGBA"
    assert _center_pixel(jpeg_bytes, (8, 8)) == (255, 255, 255)
    _assert_close(_center_pixel(jpeg_bytes, (24, 8)), (0, 0, 255))


def test_palette_png_with_transparency_composites_over_white():
    image = Image.new("P", (32, 16))
    image.putpalette([255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 6))
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), 1)
    image.info["transparency"] = 0

    jpeg_bytes = encode_refine_image_as_jpeg(image)
    assert image.mode == "P"
    assert _center_pixel(jpeg_bytes, (8, 8)) == (255, 255, 255)
    _assert_close(_center_pixel(jpeg_bytes, (24, 8)), (0, 0, 255))


def test_partial_alpha_is_blended_over_white():
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 128))
    jpeg_bytes = encode_refine_image_as_jpeg(image)
    # 128/255 black over white is mid-gray, not the raw black RGB.
    _assert_close(_center_pixel(jpeg_bytes, (8, 8)), (127, 127, 127), tol=3)


def test_gradio_and_streamlit_refine_use_the_shared_encoder():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "app.py").read_text(encoding="utf-8")
    demo_source = (root / "demo.py").read_text(encoding="utf-8")
    assert "encode_refine_image_as_jpeg(pil_img)" in app_source
    assert "encode_refine_image_as_jpeg(uploaded_image)" in demo_source
    assert "uploaded_image.save" not in demo_source


def test_rgb_upload_still_encodes_as_jpeg():
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    jpeg_bytes = encode_refine_image_as_jpeg(image)
    assert jpeg_bytes[:2] == b"\xff\xd8"
    assert _center_pixel(jpeg_bytes, (4, 4)) == (10, 20, 30)
