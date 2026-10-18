# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CLI entry point for PaperBanana Skill.
Generates publication-quality academic diagrams and plots from method text.
"""

import argparse
import asyncio
import base64
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.critic_rounds import critic_round_indexes  # noqa: E402
from utils.legacy_generation_options import (  # noqa: E402
    FIGURE_SIZE_TO_IMAGE_SIZE,
    generation_additional_info,
)

FIGURE_SIZE_CHOICES = list(FIGURE_SIZE_TO_IMAGE_SIZE)


def build_additional_info(args) -> dict:
    """The per-candidate ``additional_info``, routed through the shared helper.

    Kept as a named seam so the wiring can be tested. Building this dict inline
    was the original defect: ``figure_size`` never reached the pipeline.
    """
    return generation_additional_info(args.aspect_ratio, args.figure_size)


def ensure_model_config():
    """Copy model_config.template.yaml to model_config.yaml if missing."""
    configs_dir = PROJECT_ROOT / "configs"
    config_path = configs_dir / "model_config.yaml"
    template_path = configs_dir / "model_config.template.yaml"
    if not config_path.exists() and template_path.exists():
        shutil.copy2(template_path, config_path)


DATASET_ROOT = "PaperBananaBench"
_PREPARE_REPO_ID = "dwzhu/PaperBananaBench"
_PREPARE_SPLITS = ("test", "ref")


def dataset_task_dir(task_name: str) -> Path:
    return PROJECT_ROOT / "data" / DATASET_ROOT / task_name


def task_dir_usable(task_dir: Path) -> bool:
    """Whether a task directory is complete enough to retrieve from.

    The download short-circuit uses this. ``ref.json`` and ``images/`` are
    both required, so a half-populated directory is downloaded again.
    """
    return (task_dir / "ref.json").exists() and (task_dir / "images").is_dir()


def _prepare_script_candidates() -> tuple[Path, ...]:
    """Where ``prepare`` lives, from a checkout or an installed skill.

    ``Path(__file__)`` is used instead of ``PROJECT_ROOT`` so a test (or a
    caller) can point the data directory elsewhere without hiding the script.
    """
    here = Path(__file__).resolve().parent
    return (
        here.parent / "scripts" / "prepare_bench.py",
        here / "scripts" / "prepare_bench.py",
        here / "prepare_bench.py",
    )


def _load_prepare():
    """Return ``prepare`` from ``scripts/prepare_bench.py`` when it ships here.

    ``import scripts.prepare_bench`` is fragile: ``scripts`` is not a package,
    and an installed skill is not always launched with the repo root on
    ``sys.path``. Loading by file location works from the project root and from
    a skill tree that still contains the script. Otherwise the vendored copy
    below is used, so a skill install that did not ship ``scripts/`` can still
    download the dataset.
    """
    cached = getattr(_load_prepare, "fn", None)
    if cached is not None:
        return cached

    import importlib.util

    for script in _prepare_script_candidates():
        if not script.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "paperbanana_prepare_bench", script
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        _load_prepare.fn = module.prepare
        return module.prepare

    _load_prepare.fn = _vendored_prepare
    return _vendored_prepare


def _missing_dataset_dependency(exc: ImportError) -> bool:
    name = getattr(exc, "name", None) or ""
    if name in {"huggingface_hub", "pyarrow", "pyarrow.parquet"}:
        return True
    text = str(exc)
    missing = "No module named" in text or "cannot import name" in text
    return missing and ("huggingface_hub" in text or "pyarrow" in text)


def _vendored_diagram_record(r):
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


def _vendored_plot_record(r):
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


def _vendored_prepare(tasks, data_dir: Path, revision=None, parquet_dir=None):
    """Same behavior as ``scripts/prepare_bench.prepare``.

    Installed skills that do not include ``scripts/prepare_bench.py`` call this
    copy. Keep it in step with that script.
    """
    import pyarrow.parquet as pq

    out_root = data_dir / "PaperBananaBench"
    for task in tasks:
        task_dir = out_root / task
        (task_dir / "images").mkdir(parents=True, exist_ok=True)
        for split in _PREPARE_SPLITS:
            fname = f"data/{task}/{split}-00000-of-00001.parquet"
            if parquet_dir:
                path = Path(parquet_dir) / task / f"{split}-00000-of-00001.parquet"
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    _PREPARE_REPO_ID, fname, repo_type="dataset", revision=revision
                )
            rows = pq.read_table(path).to_pylist()
            records = []
            for r in rows:
                img = r.get("image") or {}
                if img.get("bytes"):
                    (task_dir / r["path_to_gt_image"]).write_bytes(img["bytes"])
                records.append(
                    _vendored_diagram_record(r)
                    if task == "diagram"
                    else _vendored_plot_record(r)
                )
            with open(task_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
            print(f"{task}/{split}: {len(records)} samples -> {task_dir}")


def ensure_dataset(task_name: str):
    """Download PaperBananaBench for ``task_name`` when it is not already usable.

    The Hub dataset publishes Parquet (configs ``diagram`` and ``plot``, splits
    ``test`` and ``ref``). ``prepare`` writes
    ``data/PaperBananaBench/<task>/{test,ref}.json`` and ``images/``, the layout
    the Retriever and Planner already read. Image bytes are stored on the rows
    under the original ``path_to_gt_image`` names.
    """
    task_dir = dataset_task_dir(task_name)
    if task_dir_usable(task_dir):
        return
    try:
        prepare = _load_prepare()
        prepare([task_name], PROJECT_ROOT / "data")
    except ImportError as exc:
        if _missing_dataset_dependency(exc):
            print(
                "ERROR: huggingface_hub and pyarrow are required for automatic "
                "dataset download.\n"
                "Install them with: pip install huggingface_hub pyarrow",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


def extract_final_image_b64(result: dict, exp_mode: str) -> str | None:
    """Return the base64-encoded final image from a pipeline result dict.

    Follows the same fallback order as demo.py:display_candidate_result.
    """
    task_name = "diagram"
    prefix = f"target_{task_name}"

    # Highest available critic image, including rounds past the old 0–3 cap.
    for round_idx in reversed(critic_round_indexes(result, prefix)):
        key = f"{prefix}_critic_desc{round_idx}_base64_jpg"
        if result.get(key):
            return result[key]

    # Fallback: stylist (demo_full) or planner
    if exp_mode == "demo_full":
        key = f"target_{task_name}_stylist_desc0_base64_jpg"
    else:
        key = f"target_{task_name}_desc0_base64_jpg"
    return result.get(key)


async def run(args):
    ensure_model_config()
    ensure_dataset(args.task)

    # Late imports so env is ready
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor

    # Read content from file if --content-file is given
    content = args.content
    if args.content_file:
        content = Path(args.content_file).read_text(encoding="utf-8")
    if not content:
        print("ERROR: --content or --content-file is required.", file=sys.stderr)
        sys.exit(1)

    exp_mode = args.exp_mode
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=args.retrieval_setting,
        planner_metaphor=args.planner_metaphor,
        main_model_name=args.main_model_name,
        image_gen_model_name=args.image_gen_model_name,
        work_dir=PROJECT_ROOT,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    num_candidates = args.num_candidates

    # Build data dicts
    data_list = []
    for i in range(num_candidates):
        data_list.append({
            "filename": f"skill_candidate_{i}",
            "caption": args.caption,
            "content": content,
            "visual_intent": args.caption,
            "additional_info": build_additional_info(args),
            "max_critic_rounds": args.max_critic_rounds,
        })

    # Process (parallel when multiple candidates)
    results = []
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=num_candidates, do_eval=False
    ):
        results.append(result_data)

    if not results:
        print("ERROR: Pipeline returned no results.", file=sys.stderr)
        sys.exit(1)

    # Save images
    from PIL import Image

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for idx, result in enumerate(results):
        b64 = extract_final_image_b64(result, exp_mode)
        if not b64:
            print(f"WARNING: No image produced for candidate {idx}.", file=sys.stderr)
            continue

        if "," in b64:
            b64 = b64.split(",")[1]
        image_data = base64.b64decode(b64)
        img = Image.open(BytesIO(image_data))

        if num_candidates == 1:
            save_path = output_path
        else:
            stem = output_path.stem
            suffix = output_path.suffix or ".png"
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"

        img.save(str(save_path), format="PNG")
        saved_paths.append(str(save_path))

    for p in saved_paths:
        print(p)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from main() so it can be exercised in tests."""
    parser = argparse.ArgumentParser(
        description="PaperBanana Skill: generate academic diagrams/plots from text"
    )
    parser.add_argument("--content", type=str, default="",
                        help="Method section text to visualize")
    parser.add_argument("--content-file", type=str, default="",
                        help="Path to a file containing the method section text")
    parser.add_argument("--caption", type=str, required=True,
                        help="Figure caption / visual intent")
    parser.add_argument("--task", type=str, default="diagram",
                        choices=["diagram", "plot"],
                        help="Task type: diagram or plot")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output image path (default: output.png)")
    parser.add_argument("--aspect-ratio", type=str, default="21:9",
                        choices=["21:9", "16:9", "3:2"],
                        help="Aspect ratio (default: 21:9)")
    parser.add_argument("--figure-size", type=str, default=None,
                        choices=FIGURE_SIZE_CHOICES,
                        help="Target figure width. Selects the provider image-size "
                             "tier (see FIGURE_SIZE_TO_IMAGE_SIZE). Omitted, the "
                             "provider default applies, as before.")
    parser.add_argument("--max-critic-rounds", type=int, default=3,
                        help="Max critic refinement rounds (default: 3)")
    parser.add_argument("--num-candidates", type=int, default=10,
                        help="Number of parallel candidates to generate (default: 10)")
    parser.add_argument("--retrieval-setting", type=str, default="auto",
                        choices=["auto", "manual", "random", "none"],
                        help="Retrieval mode: auto (VLM selects refs), manual, random, or none (default: auto)")
    parser.add_argument("--planner-metaphor", action="store_true",
                        help="Enable diagram-only Planner visual-metaphor discovery before detailed description output")
    parser.add_argument("--main-model-name", type=str, default="",
                        help="Main model name for VLM agents (default: from config, currently gemini-3.1-pro-preview)")
    parser.add_argument("--image-gen-model-name", type=str, default="",
                        help="Model name for image generation (default: from config, currently gemini-3.1-flash-image-preview)")
    parser.add_argument("--exp-mode", type=str, default="demo_full",
                        choices=["demo_full", "demo_planner_critic"],
                        help="Pipeline mode: demo_full (Retriever+Planner+Stylist+Visualizer+Critic) or demo_planner_critic (Retriever+Planner+Visualizer+Critic, no Stylist) (default: demo_full)")

    return parser


def main():
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
