"""Offline ``prepare`` coverage.

Builds tiny Parquet files locally and checks the JSON layout plus the image
bytes. Nothing here contacts the Hub.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_bench import prepare
from skill.run import _vendored_prepare


IMAGE_TYPE = pa.struct([
    pa.field("bytes", pa.binary()),
    pa.field("path", pa.string()),
])

DIAGRAM_TEST_BYTES = b"diagram-test-jpeg"
DIAGRAM_REF_BYTES = b"diagram-ref-jpeg"
PLOT_TEST_BYTES = b"plot-test-jpeg"
PLOT_REF_BYTES = b"plot-ref-jpeg"


def _write_split(parquet_dir: Path, task: str, split: str, row: dict) -> None:
    path = parquet_dir / task / f"{split}-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    names = []
    for key, value in row.items():
        names.append(key)
        if key == "image":
            arrays.append(pa.array([value], type=IMAGE_TYPE))
        else:
            arrays.append(pa.array([value]))
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


def _write_fixture(parquet_dir: Path) -> None:
    _write_split(parquet_dir, "diagram", "test", {
        "id": "diagram-test",
        "content": "Encoder–decoder with a café",
        "visual_intent": "Show the pipeline",
        "path_to_gt_image": "images/test.jpg",
        "category": "architecture",
        "source_pdf": "paper.pdf",
        "diagram_page_num": 4,
        "method_section_title": "3 Method",
        "width": 800,
        "height": 600,
        "rounded_ratio": "4:3",
        "split": "test",
        "image": {"bytes": DIAGRAM_TEST_BYTES, "path": None},
    })
    _write_split(parquet_dir, "diagram", "ref", {
        "id": "diagram-ref",
        "content": "Reference café",
        "visual_intent": "Reference pipeline",
        "path_to_gt_image": "images/ref.jpg",
        "category": "architecture",
        "source_pdf": "paper.pdf",
        "diagram_page_num": 2,
        "method_section_title": "3 Method",
        "width": 640,
        "height": 480,
        "rounded_ratio": "4:3",
        "split": "ref",
        "image": {"bytes": DIAGRAM_REF_BYTES, "path": None},
    })
    _write_split(parquet_dir, "plot", "test", {
        "id": "plot-test",
        "content": json.dumps({"x": [1, 2], "y": [3, 4]}),
        "visual_intent": "Accuracy vs. steps",
        "path_to_gt_image": "images/test.jpg",
        "original_category": "line_chart",
        "gt_code": "plt.plot([1, 2], [3, 4])",
        "width": 640,
        "height": 480,
        "rounded_ratio": "4:3",
        "difficulty": "medium",
        "category": "line",
        "image": {"bytes": PLOT_TEST_BYTES, "path": None},
    })
    _write_split(parquet_dir, "plot", "ref", {
        "id": "plot-ref",
        "content": json.dumps({"x": [0], "y": [1]}),
        "visual_intent": "Reference curve",
        "path_to_gt_image": "images/ref.jpg",
        "original_category": "line_chart",
        "gt_code": "plt.plot([0], [1])",
        "width": 320,
        "height": 240,
        "rounded_ratio": "4:3",
        "difficulty": "easy",
        "category": "line",
        "image": {"bytes": PLOT_REF_BYTES, "path": None},
    })


def _assert_task_layout(data_dir: Path) -> None:
    root = data_dir / "PaperBananaBench"

    expected_diagram_test = {
        "id": "diagram-test",
        "content": "Encoder–decoder with a café",
        "visual_intent": "Show the pipeline",
        "path_to_gt_image": "images/test.jpg",
        "category": "architecture",
        "additional_info": {
            "file_name": "paper.pdf",
            "file_path": "pdfs/paper.pdf",
            "diagram_page_num": 4,
            "method_section_title": "3 Method",
            "width": 800,
            "height": 600,
            "rounded_ratio": "4:3",
        },
        "split": "test",
    }
    expected_diagram_ref = {
        "id": "diagram-ref",
        "content": "Reference café",
        "visual_intent": "Reference pipeline",
        "path_to_gt_image": "images/ref.jpg",
        "category": "architecture",
        "additional_info": {
            "file_name": "paper.pdf",
            "file_path": "pdfs/paper.pdf",
            "diagram_page_num": 2,
            "method_section_title": "3 Method",
            "width": 640,
            "height": 480,
            "rounded_ratio": "4:3",
        },
        "split": "ref",
    }
    for path, expected in (
        (root / "diagram" / "test.json", expected_diagram_test),
        (root / "diagram" / "ref.json", expected_diagram_ref),
    ):
        raw = path.read_text(encoding="utf-8")
        assert raw == json.dumps([expected], ensure_ascii=False, indent=4)

    diagram_images = root / "diagram" / "images"
    assert (diagram_images / "test.jpg").read_bytes() == DIAGRAM_TEST_BYTES
    assert (diagram_images / "ref.jpg").read_bytes() == DIAGRAM_REF_BYTES

    expected_plot_test = {
        "id": "plot-test",
        "content": {"x": [1, 2], "y": [3, 4]},
        "visual_intent": "Accuracy vs. steps",
        "path_to_gt_image": "images/test.jpg",
        "original_category": "line_chart",
        "additional_info": {
            "gt_code": "plt.plot([1, 2], [3, 4])",
            "width": 640.0,
            "height": 480.0,
            "rounded_ratio": "4:3",
        },
        "difficulty": "medium",
        "category": "line",
    }
    expected_plot_ref = {
        "id": "plot-ref",
        "content": {"x": [0], "y": [1]},
        "visual_intent": "Reference curve",
        "path_to_gt_image": "images/ref.jpg",
        "original_category": "line_chart",
        "additional_info": {
            "gt_code": "plt.plot([0], [1])",
            "width": 320.0,
            "height": 240.0,
            "rounded_ratio": "4:3",
        },
        "difficulty": "easy",
        "category": "line",
    }
    for path, expected in (
        (root / "plot" / "test.json", expected_plot_test),
        (root / "plot" / "ref.json", expected_plot_ref),
    ):
        raw = path.read_text(encoding="utf-8")
        assert raw == json.dumps([expected], ensure_ascii=False, indent=4)

    plot_images = root / "plot" / "images"
    assert (plot_images / "test.jpg").read_bytes() == PLOT_TEST_BYTES
    assert (plot_images / "ref.jpg").read_bytes() == PLOT_REF_BYTES


def _run_prepare(prepare_fn, tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    prepare_fn(["diagram", "plot"], tmp_path / "data", parquet_dir=tmp_path)
    _assert_task_layout(tmp_path / "data")


def test_prepare_writes_json_schema_and_image(tmp_path):
    _run_prepare(prepare, tmp_path)


def test_vendored_prepare_writes_the_same_layout(tmp_path):
    _run_prepare(_vendored_prepare, tmp_path)
