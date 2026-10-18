#!/usr/bin/env python3
"""Download PaperBananaBench from the Hugging Face Hub (parquet) and write the
layout the PaperBanana code expects:

    data/PaperBananaBench/{diagram,plot}/{test,ref}.json
    data/PaperBananaBench/{diagram,plot}/images/<file>.jpg

Usage:  python scripts/prepare_bench.py [--tasks diagram plot] [--data-dir data]
Requires: huggingface_hub, pyarrow
"""
import argparse
import json
from pathlib import Path

REPO_ID = "dwzhu/PaperBananaBench"
SPLITS = ("test", "ref")


def _diagram_record(r):
    return {
        "id": r["id"],
        "content": r["content"],
        "visual_intent": r["visual_intent"],
        "path_to_gt_image": r["path_to_gt_image"],
        "category": r["category"],
        "additional_info": {
            "file_name": r["source_pdf"],
            "file_path": f"pdfs/{r['source_pdf']}",
            "diagram_page_num": r["diagram_page_num"],
            "method_section_title": r["method_section_title"],
            "width": r["width"],
            "height": r["height"],
            "rounded_ratio": r["rounded_ratio"],
        },
        "split": r["split"],
    }


def _plot_record(r):
    return {
        "id": r["id"],
        "content": json.loads(r["content"]),
        "visual_intent": r["visual_intent"],
        "path_to_gt_image": r["path_to_gt_image"],
        "original_category": r["original_category"],
        "additional_info": {
            "gt_code": r["gt_code"],
            "width": float(r["width"]),
            "height": float(r["height"]),
            "rounded_ratio": r["rounded_ratio"],
        },
        "difficulty": r["difficulty"],
        "category": r["category"],
    }


def prepare(tasks, data_dir: Path, revision=None, parquet_dir=None):
    import pyarrow.parquet as pq
    out_root = data_dir / "PaperBananaBench"
    for task in tasks:
        task_dir = out_root / task
        (task_dir / "images").mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            fname = f"data/{task}/{split}-00000-of-00001.parquet"
            if parquet_dir:
                path = Path(parquet_dir) / task / f"{split}-00000-of-00001.parquet"
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(REPO_ID, fname, repo_type="dataset", revision=revision)
            rows = pq.read_table(path).to_pylist()
            records = []
            for r in rows:
                img = r.get("image") or {}
                if img.get("bytes"):
                    (task_dir / r["path_to_gt_image"]).write_bytes(img["bytes"])
                records.append(_diagram_record(r) if task == "diagram" else _plot_record(r))
            with open(task_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
            print(f"{task}/{split}: {len(records)} samples -> {task_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["diagram", "plot"], choices=["diagram", "plot"])
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--parquet-dir", default=None, help=argparse.SUPPRESS)  # local testing
    a = ap.parse_args()
    prepare(a.tasks, Path(a.data_dir), a.revision, a.parquet_dir)
