"""Critic-round controls adapted from PR #73.

The Critic does not load a style guide. These tests cover only
``max_critic_rounds`` on the planner-critic path.
"""

import asyncio

from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor


def _config(tmp_path, **kwargs):
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


def test_sample_zero_max_critic_rounds_overrides_positive_config(tmp_path):
    exp_config = _config(tmp_path, max_critic_rounds=3, exp_mode="dev_planner_critic")

    class FailCritic:
        async def process(self, data, source="stylist"):
            raise AssertionError("sample max_critic_rounds=0 should override a positive config")

    processor = _planner_critic_processor(exp_config, FailCritic())
    result = asyncio.run(
        processor.process_single_query(
            {
                "content": "method content",
                "visual_intent": "figure caption",
                "max_critic_rounds": 0,
            },
            do_eval=False,
        )
    )

    assert result["eval_image_field"] == "target_diagram_desc0_base64_jpg"
    assert "current_critic_round" not in result
