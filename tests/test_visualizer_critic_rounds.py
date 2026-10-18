"""Visualizer critic-round limit adapted from PR #56.

The Refine UI allows up to 5 critic rounds and the processor loops
``max_rounds``, but the visualizer used to render only ``range(3)``.
"""

import asyncio
from unittest.mock import patch

from agents.visualizer_agent import VisualizerAgent
from utils.config import ExpConfig


def _agent(tmp_path, max_critic_rounds=3):
    exp_config = ExpConfig(
        dataset_name="PaperBananaBench",
        task_name="diagram",
        exp_mode="dev_planner_critic",
        main_model_name="gemini-2.5-pro",
        image_gen_model_name="gemini-2.5-flash-image-preview",
        work_dir=tmp_path,
        timestamp="test",
        max_critic_rounds=max_critic_rounds,
    )
    return VisualizerAgent(exp_config=exp_config)


_OMIT = object()


def _critic_data(rounds=5, max_critic_rounds=_OMIT):
    data = {
        f"target_diagram_critic_desc{round_idx}": f"desc-{round_idx}"
        for round_idx in range(rounds)
    }
    if max_critic_rounds is not _OMIT:
        data["max_critic_rounds"] = max_critic_rounds
    return data


def _render(agent, data):
    async def fake_dispatch(*args, **kwargs):
        return ["png-b64"]

    with patch(
        "agents.visualizer_agent.generation_utils.dispatch_image_generation",
        side_effect=fake_dispatch,
    ) as dispatch, patch(
        "agents.visualizer_agent.image_utils.convert_png_b64_to_jpg_b64",
        return_value="jpg-b64",
    ):
        result = asyncio.run(agent.process(data))
    return result, dispatch.call_count


def _rendered_rounds(result):
    return [
        round_idx
        for round_idx in range(5)
        if result.get(f"target_diagram_critic_desc{round_idx}_base64_jpg") == "jpg-b64"
    ]


def test_default_three_rounds_skips_later_critic_descriptions(tmp_path):
    result, calls = _render(_agent(tmp_path), _critic_data())

    assert _rendered_rounds(result) == [0, 1, 2]
    assert calls == 3
    assert "target_diagram_critic_desc3_base64_jpg" not in result
    assert "target_diagram_critic_desc4_base64_jpg" not in result


def test_sample_max_critic_rounds_renders_rounds_four_and_five(tmp_path):
    result, calls = _render(
        _agent(tmp_path, max_critic_rounds=3),
        _critic_data(max_critic_rounds=5),
    )

    assert _rendered_rounds(result) == [0, 1, 2, 3, 4]
    assert calls == 5


def test_config_max_critic_rounds_used_when_sample_omits_it(tmp_path):
    result, calls = _render(_agent(tmp_path, max_critic_rounds=5), _critic_data())

    assert _rendered_rounds(result) == [0, 1, 2, 3, 4]
    assert calls == 5


def test_sample_max_of_three_matches_historical_limit(tmp_path):
    result, calls = _render(
        _agent(tmp_path, max_critic_rounds=5),
        _critic_data(max_critic_rounds=3),
    )

    assert _rendered_rounds(result) == [0, 1, 2]
    assert calls == 3


def test_zero_sample_renders_no_critic_rounds(tmp_path):
    result, calls = _render(
        _agent(tmp_path, max_critic_rounds=5),
        _critic_data(max_critic_rounds=0),
    )

    assert _rendered_rounds(result) == []
    assert calls == 0


def test_negative_sample_clamps_to_zero(tmp_path):
    result, calls = _render(
        _agent(tmp_path, max_critic_rounds=5),
        _critic_data(max_critic_rounds=-2),
    )

    assert _rendered_rounds(result) == []
    assert calls == 0


def test_none_sample_uses_config_instead_of_default_three(tmp_path):
    result, calls = _render(
        _agent(tmp_path, max_critic_rounds=5),
        _critic_data(max_critic_rounds=None),
    )

    assert _rendered_rounds(result) == [0, 1, 2, 3, 4]
    assert calls == 5


def test_blank_and_non_numeric_samples_fall_through_to_config(tmp_path):
    for sample_value in ("", "five"):
        result, calls = _render(
            _agent(tmp_path, max_critic_rounds=5),
            _critic_data(max_critic_rounds=sample_value),
        )

        assert _rendered_rounds(result) == [0, 1, 2, 3, 4], sample_value
        assert calls == 5


def test_integer_string_sample_is_honored(tmp_path):
    result, calls = _render(
        _agent(tmp_path, max_critic_rounds=5),
        _critic_data(max_critic_rounds="4"),
    )

    assert _rendered_rounds(result) == [0, 1, 2, 3]
    assert calls == 4
    assert "target_diagram_critic_desc4_base64_jpg" not in result
