"""Style-guide loading and critic-round controls adapted from PR #73.

Agentic Gemini code-execution coverage from that PR is intentionally omitted.
"""

import asyncio

from agents.critic_agent import CriticAgent
from agents.polish_agent import PolishAgent
from agents.stylist_agent import StylistAgent
from utils import generation_utils
from utils.config import ExpConfig, load_style_guide_for_task, style_guide_filename_for_task
from utils.paperviz_processor import PaperVizProcessor


def _write_style_guides(work_dir):
    style_dir = work_dir / "style_guides"
    style_dir.mkdir(parents=True)
    (style_dir / "neurips2025_diagram_style_guide.md").write_text(
        "DIAGRAM_STYLE_GUIDE_SENTINEL",
        encoding="utf-8",
    )
    (style_dir / "neurips2025_plot_style_guide.md").write_text(
        "PLOT_STYLE_GUIDE_SENTINEL",
        encoding="utf-8",
    )


def _diagram_data():
    return {
        "target_diagram_desc0": "Initial planner description",
        "target_diagram_desc0_base64_jpg": "",
        "content": "method content",
        "visual_intent": "figure caption",
    }


def _config(tmp_path, **kwargs):
    _write_style_guides(tmp_path)
    defaults = {
        "dataset_name": "PaperBananaBench",
        "task_name": "diagram",
        "exp_mode": "demo_planner_critic",
        "main_model_name": "gemini-2.5-pro",
        "image_gen_model_name": "gemini-2.5-flash-image-preview",
        "work_dir": tmp_path,
        "timestamp": "test",
    }
    defaults.update(kwargs)
    return ExpConfig(**defaults)


def test_style_guide_filename_normalizes_task_aliases():
    assert style_guide_filename_for_task("diagram") == "neurips2025_diagram_style_guide.md"
    assert style_guide_filename_for_task("plots") == "neurips2025_plot_style_guide.md"
    assert style_guide_filename_for_task("statistical plot") == "neurips2025_plot_style_guide.md"


def test_load_style_guide_for_task_reads_matching_file(tmp_path):
    _write_style_guides(tmp_path)
    assert load_style_guide_for_task(tmp_path, "diagram") == "DIAGRAM_STYLE_GUIDE_SENTINEL"
    assert load_style_guide_for_task(tmp_path, "plots") == "PLOT_STYLE_GUIDE_SENTINEL"


def test_default_critic_path_uses_router_and_loads_task_style_guide(tmp_path, monkeypatch):
    exp_config = _config(tmp_path)
    captured = {}

    async def fake_router(model_name, contents, config, max_attempts, retry_delay, error_context=""):
        captured["model_name"] = model_name
        captured["contents"] = contents
        return [
            '{"critic_suggestions":"No changes needed.","revised_description":"No changes needed."}'
        ]

    monkeypatch.setattr(generation_utils, "call_model_with_retry_async", fake_router)

    result = asyncio.run(CriticAgent(exp_config=exp_config).process(_diagram_data(), source="planner"))

    assert captured["model_name"] == "gemini-2.5-pro"
    assert "DIAGRAM_STYLE_GUIDE_SENTINEL" in captured["contents"][-1]["text"]
    assert result["target_diagram_critic_suggestions0"] == "No changes needed."
    assert result["target_diagram_critic_desc0"] == "Initial planner description"


def test_stylist_loads_task_specific_style_guide(tmp_path, monkeypatch):
    exp_config = _config(tmp_path, task_name="plot")
    captured = {}

    async def fake_router(model_name, contents, config, max_attempts, retry_delay, error_context=""):
        captured["prompt"] = contents[0]["text"]
        return ["Styled plot description"]

    monkeypatch.setattr(generation_utils, "call_model_with_retry_async", fake_router)
    data = {
        "target_plot_desc0": "Initial plot description",
        "content": {"x": [1, 2], "y": [3, 4]},
        "visual_intent": "show trend",
    }

    result = asyncio.run(StylistAgent(exp_config=exp_config).process(data))

    assert "PLOT_STYLE_GUIDE_SENTINEL" in captured["prompt"]
    assert "DIAGRAM_STYLE_GUIDE_SENTINEL" not in captured["prompt"]
    assert result["target_plot_stylist_desc0"] == "Styled plot description"


def test_polish_loads_task_specific_style_guide(tmp_path, monkeypatch):
    exp_config = _config(tmp_path, task_name="plot", exp_mode="dev_polish")
    image_dir = tmp_path / "data" / "PaperBananaBench" / "plot"
    image_dir.mkdir(parents=True)
    (image_dir / "gt.jpg").write_bytes(b"fake-image")
    captured = {}

    async def fake_router(model_name, contents, config, max_attempts, retry_delay, error_context=""):
        captured["prompt"] = contents[0]["text"]
        return ["No changes needed"]

    async def fake_image(*args, **kwargs):
        captured["image_called"] = True
        return [None]

    monkeypatch.setattr(generation_utils, "call_model_with_retry_async", fake_router)
    monkeypatch.setattr(generation_utils, "call_gemini_with_retry_async", fake_image)
    monkeypatch.setattr(generation_utils, "openrouter_client", None)

    asyncio.run(PolishAgent(exp_config=exp_config).process({"path_to_gt_image": "gt.jpg"}))

    assert "PLOT_STYLE_GUIDE_SENTINEL" in captured["prompt"]
    assert "DIAGRAM_STYLE_GUIDE_SENTINEL" not in captured["prompt"]


def test_zero_critic_rounds_bypasses_critic_and_keeps_planner_eval_image(tmp_path):
    exp_config = _config(tmp_path, max_critic_rounds=0)

    class FailCritic:
        async def process(self, data, source="stylist"):
            raise AssertionError("critic should not run when max_rounds is 0")

    class FailVisualizer:
        async def process(self, data):
            raise AssertionError("visualizer should not rerun when max_rounds is 0")

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=None,
        planner_agent=None,
        visualizer_agent=FailVisualizer(),
        stylist_agent=None,
        critic_agent=FailCritic(),
        retriever_agent=None,
        polish_agent=None,
    )
    data = {"target_diagram_desc0_base64_jpg": "planner-image"}

    result = asyncio.run(
        processor._run_critic_iterations(data, "diagram", max_rounds=0, source="planner")
    )

    assert result["eval_image_field"] == "target_diagram_desc0_base64_jpg"
    assert "current_critic_round" not in result


def _planner_critic_processor(exp_config, critic):
    class FakeRetriever:
        async def process(self, data, retrieval_setting="auto"):
            data["top10_references"] = []
            return data

    class FakePlanner:
        async def process(self, data):
            data["target_diagram_desc0"] = "planned description"
            return data

    class FakeVisualizer:
        async def process(self, data):
            round_idx = data.get("current_critic_round")
            if round_idx is None:
                data["target_diagram_desc0_base64_jpg"] = "planner-image"
            else:
                data[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = f"critic-image-{round_idx}"
            return data

    return PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=None,
        planner_agent=FakePlanner(),
        visualizer_agent=FakeVisualizer(),
        stylist_agent=None,
        critic_agent=critic,
        retriever_agent=FakeRetriever(),
        polish_agent=None,
    )


def test_zero_critic_rounds_from_config_bypasses_planner_critic_mode(tmp_path):
    exp_config = _config(tmp_path, max_critic_rounds=0)

    class FailCritic:
        async def process(self, data, source="stylist"):
            raise AssertionError("critic should not run when config max_critic_rounds is 0")

    processor = _planner_critic_processor(exp_config, FailCritic())
    data = {"content": "method content", "visual_intent": "figure caption"}

    result = asyncio.run(processor.process_single_query(data, do_eval=False))

    assert result["eval_image_field"] == "target_diagram_desc0_base64_jpg"
    assert "current_critic_round" not in result


def test_planner_critic_honors_config_max_critic_rounds(tmp_path):
    """CLI --max_critic_rounds is stored on ExpConfig and must drive this path."""
    exp_config = _config(tmp_path, max_critic_rounds=1, exp_mode="dev_planner_critic")
    calls = {"critic": 0, "source": None}

    class CountingCritic:
        async def process(self, data, source="stylist"):
            calls["critic"] += 1
            calls["source"] = source
            round_idx = data["current_critic_round"]
            data[f"target_diagram_critic_suggestions{round_idx}"] = "Keep revising."
            data[f"target_diagram_critic_desc{round_idx}"] = f"revised {round_idx}"
            return data

    processor = _planner_critic_processor(exp_config, CountingCritic())
    result = asyncio.run(
        processor.process_single_query(
            {"content": "method content", "visual_intent": "figure caption"},
            do_eval=False,
        )
    )

    assert calls == {"critic": 1, "source": "planner"}
    assert result["eval_image_field"] == "target_diagram_critic_desc0_base64_jpg"


def test_sample_max_critic_rounds_overrides_config(tmp_path):
    exp_config = _config(tmp_path, max_critic_rounds=3, exp_mode="dev_planner_critic")
    calls = {"critic": 0}

    class CountingCritic:
        async def process(self, data, source="stylist"):
            calls["critic"] += 1
            round_idx = data["current_critic_round"]
            data[f"target_diagram_critic_suggestions{round_idx}"] = "Keep revising."
            data[f"target_diagram_critic_desc{round_idx}"] = f"revised {round_idx}"
            return data

    processor = _planner_critic_processor(exp_config, CountingCritic())
    asyncio.run(
        processor.process_single_query(
            {
                "content": "method content",
                "visual_intent": "figure caption",
                "max_critic_rounds": 2,
            },
            do_eval=False,
        )
    )

    assert calls["critic"] == 2
