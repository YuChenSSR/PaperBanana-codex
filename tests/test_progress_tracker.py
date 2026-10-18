"""Per-stage progress tracking. Adapted from #56 by @noking-xdj.

These tests cover the tracker itself and its wiring through the batch pipeline.
They do not change critic-round limits, prompts, or failure isolation.
"""

import asyncio
from unittest import mock

from utils.config import ExpConfig
from utils.paperviz_processor import (
    PaperVizProcessor,
    ProgressTracker,
    generation_progress_update,
    pipeline_stages,
)


def test_pipeline_stage_names_follow_mode_and_round_budget():
    assert pipeline_stages("vanilla") == ["Processing"]
    assert pipeline_stages("dev_polish") == ["Processing"]
    assert pipeline_stages("dev_retriever") == ["Processing"]
    assert pipeline_stages("dev_planner", 5) == ["Retriever", "Planner", "Visualizer"]
    assert pipeline_stages("dev_planner_stylist") == [
        "Retriever",
        "Planner",
        "Stylist",
        "Visualizer",
    ]
    assert pipeline_stages("demo_planner_critic", 0) == [
        "Retriever",
        "Planner",
        "Visualizer",
    ]
    assert pipeline_stages("demo_full", 1) == [
        "Retriever",
        "Planner",
        "Stylist",
        "Visualizer",
        "Critic R0",
        "Visualizer (R0)",
    ]
    assert pipeline_stages("demo_full", "nope") == [
        "Retriever",
        "Planner",
        "Stylist",
        "Visualizer",
    ]


def test_negative_placeholder_ids_do_not_affect_stage_counts():
    """A -1 sentinel must not tie or stick the active stage on Retriever."""
    events = []
    tracker = ProgressTracker(
        {0: ["Retriever", "Planner"]},
        callback=events.append,
    )

    tracker.enter_stage(-1, "Retriever", "sentinel-model")
    tracker.complete_stage(-1, "Retriever")
    assert events == []
    assert -1 not in tracker.expected_stages

    tracker.enter_stage(0, "Planner", "real-model")
    tracker.enter_stage(-1, "Retriever", "sentinel-model")
    tracker.complete_stage(-1, "Planner")
    status = tracker.get_status()

    assert status["stage"] == "Planner"
    assert status["model"] == "real-model"
    assert status["overall_pct"] == 0
    assert status["stage_done"] == 0
    assert status["all_stage_done"]["Retriever"] == 0


def test_tied_stages_prefer_the_later_pipeline_stage():
    tracker = ProgressTracker(
        {
            0: pipeline_stages("demo_planner_critic", 1),
            1: pipeline_stages("demo_planner_critic", 2),
        }
    )
    tracker.enter_stage(0, "Visualizer (R0)")
    tracker.enter_stage(1, "Critic R1")
    assert tracker.get_status()["stage"] == "Critic R1"


def test_overall_progress_uses_each_candidates_critic_rounds():
    short = pipeline_stages("demo_planner_critic", 1)
    long = pipeline_stages("demo_planner_critic", 3)
    assert len(short) == 5
    assert len(long) == 9
    assert "Critic R2" not in short
    assert "Critic R2" in long

    tracker = ProgressTracker({0: short, 1: long})
    for stage in short:
        tracker.enter_stage(0, stage)
        tracker.complete_stage(0, stage)
    for stage in long[:5]:
        tracker.enter_stage(1, stage)
        tracker.complete_stage(1, stage)

    # 10/14. Sizing the plan from candidate 0 only would already report 10/10.
    assert tracker.get_status()["overall_pct"] == 10 / 14
    assert tracker.get_status()["all_stage_done"]["Critic R2"] == 0

    tracker.complete_stage(0, "Critic R2")
    assert tracker.get_status()["all_stage_done"]["Critic R2"] == 0

    for stage in long[5:]:
        tracker.enter_stage(1, stage)
        tracker.complete_stage(1, stage)
    tracker.complete_stage(1, long[-1])

    status = tracker.get_status()
    assert status["overall_pct"] == 1
    assert status["stage"] == "Visualizer (R2)"
    assert status["stage_done"] == 1
    assert status["total"] == 1
    assert status["all_stage_done"]["Critic R0"] == 2
    assert status["all_stage_done"]["Critic R2"] == 1


def test_from_batch_plans_every_candidate_and_drops_placeholder_ids():
    tracker = ProgressTracker.from_batch(
        "demo_planner_critic",
        [
            {"candidate_id": -1, "max_critic_rounds": 1},
            {"candidate_id": 5, "max_critic_rounds": 3},
            {},
        ],
        default_rounds=4,
    )

    assert list(tracker.expected_stages) == [0, 1, 2]
    assert tracker.expected_stages[0] == pipeline_stages("demo_planner_critic", 1)
    assert tracker.expected_stages[1] == pipeline_stages("demo_planner_critic", 3)
    # The candidate with no budget must use the default, not candidate 0's budget.
    assert tracker.expected_stages[2] == pipeline_stages("demo_planner_critic", 4)
    assert -1 not in tracker.expected_stages


def test_generation_progress_update_stays_inside_generation_band():
    fraction, desc = generation_progress_update(
        {
            "overall_pct": 0.5,
            "stage": "Planner",
            "stage_done": 1,
            "total": 2,
            "model": "gemini",
        }
    )
    assert fraction == 0.5
    assert desc == "Planner (1/2) [gemini]"

    fraction, _desc = generation_progress_update(
        {"overall_pct": 1, "stage": "Visualizer", "stage_done": 2, "total": 2}
    )
    assert fraction == 0.9

    fraction, desc = generation_progress_update({})
    assert fraction == 0.1
    assert desc == "Working (0/0)"


def _config(tmp_path, **kwargs):
    defaults = {
        "dataset_name": "PaperBananaBench",
        "task_name": "diagram",
        "exp_mode": "demo_planner_critic",
        "main_model_name": "text-model",
        "image_gen_model_name": "image-model",
        "work_dir": tmp_path,
        "timestamp": "test",
        "max_critic_rounds": 3,
    }
    defaults.update(kwargs)
    return ExpConfig(**defaults)


def _processor(exp_config, critic, planner=None, visualizer=None, stylist=None):
    class Retriever:
        async def process(self, data, retrieval_setting="auto"):
            data["top10_references"] = ["ref"]
            data["retrieved_examples"] = []
            return data

    class Planner:
        async def process(self, data):
            data["target_diagram_desc0"] = "plan"
            return data

    class Stylist:
        async def process(self, data):
            data["target_diagram_stylist_desc0"] = "style"
            return data

    class Visualizer:
        async def process(self, data):
            round_idx = data.get("current_critic_round")
            if round_idx is None:
                if exp_config.exp_mode in ("dev_full", "demo_full", "dev_planner_stylist"):
                    data["target_diagram_stylist_desc0_base64_jpg"] = "stylist-image"
                else:
                    data["target_diagram_desc0_base64_jpg"] = "planner-image"
            else:
                data[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = (
                    f"critic-image-{round_idx}"
                )
            return data

    return PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=None,
        planner_agent=planner or Planner(),
        visualizer_agent=visualizer or Visualizer(),
        stylist_agent=stylist or Stylist(),
        critic_agent=critic,
        retriever_agent=Retriever(),
        polish_agent=None,
    )


def _distinct_stages(statuses):
    distinct = []
    for status in statuses:
        stage = status["stage"]
        if not distinct or distinct[-1] != stage:
            distinct.append(stage)
    return distinct


def test_batch_tracks_each_candidates_rounds_and_ignores_placeholder_ids(tmp_path):
    class RevisingCritic:
        def __init__(self):
            self.calls = []

        async def process(self, data, source="stylist"):
            round_idx = data["current_critic_round"]
            self.calls.append((data.get("candidate_id"), round_idx, source))
            data[f"target_diagram_critic_suggestions{round_idx}"] = "Keep revising."
            data[f"target_diagram_critic_desc{round_idx}"] = f"revised {round_idx}"
            return data

    critic = RevisingCritic()
    processor = _processor(
        _config(tmp_path, exp_mode="demo_planner_critic", max_critic_rounds=1),
        critic,
    )
    entered = []
    original_enter = ProgressTracker.enter_stage

    def spy(self, candidate_id, stage, model_name=""):
        entered.append(candidate_id)
        return original_enter(self, candidate_id, stage, model_name)

    async def collect():
        statuses = []
        results = []
        async for item in processor.process_queries_batch(
            [
                {"candidate_id": -1, "max_critic_rounds": 1, "content": "short"},
                {"candidate_id": 5, "max_critic_rounds": 3, "content": "long"},
            ],
            max_concurrent=2,
            do_eval=False,
            progress_callback=statuses.append,
        ):
            results.append(item)
        return statuses, results

    with mock.patch.object(ProgressTracker, "enter_stage", spy):
        statuses, results = asyncio.run(collect())

    assert len(results) == 2
    assert -1 not in entered
    assert set(entered) <= {0, 1}
    assert [call[1] for call in critic.calls if call[0] == -1] == [0]
    assert [call[1] for call in critic.calls if call[0] == 5] == [0, 1, 2]
    assert {call[2] for call in critic.calls} == {"planner"}
    assert statuses[-1]["overall_pct"] == 1
    assert statuses[-1]["all_stage_done"]["Critic R0"] == 2
    assert statuses[-1]["all_stage_done"]["Critic R2"] == 1
    assert any(
        status["stage"] == "Critic R2"
        and status["all_stage_done"]["Critic R2"] == 0
        and status["overall_pct"] < 1
        for status in statuses
    )


def test_full_pipeline_reports_stages_in_order(tmp_path):
    class RevisingCritic:
        async def process(self, data, source="stylist"):
            round_idx = data["current_critic_round"]
            data[f"target_diagram_critic_suggestions{round_idx}"] = "Keep revising."
            data[f"target_diagram_critic_desc{round_idx}"] = "revised"
            return data

    processor = _processor(
        _config(tmp_path, exp_mode="demo_full", max_critic_rounds=1),
        RevisingCritic(),
    )

    async def collect():
        statuses = []
        async for _item in processor.process_queries_batch(
            [{"candidate_id": 0, "max_critic_rounds": 1, "content": "method"}],
            do_eval=False,
            progress_callback=statuses.append,
        ):
            pass
        return statuses

    statuses = asyncio.run(collect())
    assert _distinct_stages(statuses) == [
        "Retriever",
        "Planner",
        "Stylist",
        "Visualizer",
        "Critic R0",
        "Visualizer (R0)",
    ]
    assert statuses[-1]["overall_pct"] == 1
    assert statuses[-1]["model"] == "image-model"


def test_early_stop_counts_only_that_candidates_remaining_rounds(tmp_path):
    class StopCritic:
        def __init__(self):
            self.rounds = []

        async def process(self, data, source="stylist"):
            round_idx = data["current_critic_round"]
            self.rounds.append((data.get("candidate_id"), round_idx))
            data[f"target_diagram_critic_suggestions{round_idx}"] = "No changes needed."
            return data

    class CountingVisualizer:
        def __init__(self):
            self.calls = 0

        async def process(self, data):
            self.calls += 1
            round_idx = data.get("current_critic_round")
            if round_idx is None:
                data["target_diagram_desc0_base64_jpg"] = "planner-image"
            else:
                data[f"target_diagram_critic_desc{round_idx}_base64_jpg"] = "should-not-run"
            return data

    critic = StopCritic()
    visualizer = CountingVisualizer()
    processor = _processor(
        _config(tmp_path, exp_mode="demo_planner_critic", max_critic_rounds=3),
        critic,
        visualizer=visualizer,
    )

    async def collect():
        statuses = []
        async for _item in processor.process_queries_batch(
            [
                {"candidate_id": "stop-early", "max_critic_rounds": 3},
                {"candidate_id": "also-stops", "max_critic_rounds": 1},
            ],
            max_concurrent=2,
            do_eval=False,
            progress_callback=statuses.append,
        ):
            pass
        return statuses

    statuses = asyncio.run(collect())

    assert sorted(critic.rounds) == [("also-stops", 0), ("stop-early", 0)]
    assert visualizer.calls == 2
    final = statuses[-1]
    assert final["overall_pct"] == 1
    assert final["all_stage_done"]["Critic R0"] == 2
    assert final["all_stage_done"]["Critic R2"] == 1
    assert final["all_stage_done"]["Visualizer (R2)"] == 1


def test_candidate_failure_does_not_cancel_the_others(tmp_path):
    release = asyncio.Event()
    state = {"cancelled": False, "finished": False, "planner_calls": []}

    class FailingPlanner:
        async def process(self, data):
            candidate_id = data["candidate_id"]
            state["planner_calls"].append(candidate_id)
            if candidate_id == "fail":
                release.set()
                raise RuntimeError("candidate failed")
            try:
                await release.wait()
                await asyncio.sleep(0.05)
                data["target_diagram_desc0"] = "ok"
                state["finished"] = True
                return data
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

    class IdleCritic:
        async def process(self, data, source="stylist"):
            raise AssertionError("critic should not run when max_critic_rounds is 0")

    processor = _processor(
        _config(tmp_path, exp_mode="dev_planner_critic", max_critic_rounds=0),
        IdleCritic(),
        planner=FailingPlanner(),
    )

    async def consume():
        async for _item in processor.process_queries_batch(
            [
                {"candidate_id": "fail", "max_critic_rounds": 0},
                {"candidate_id": "ok", "max_critic_rounds": 0},
            ],
            max_concurrent=2,
            do_eval=False,
            progress_callback=lambda _status: None,
        ):
            pass

    async def main():
        try:
            await consume()
        except RuntimeError as exc:
            assert str(exc) == "candidate failed"
            await asyncio.sleep(0.2)
        return state

    result = asyncio.run(main())
    assert result["planner_calls"].count("ok") == 1
    assert result["finished"] is True
    assert result["cancelled"] is False


def test_process_parallel_candidates_forwards_progress_callback(monkeypatch):
    import app as gradio_app

    seen = {}

    async def fake_batch(self, data_list, max_concurrent=50, do_eval=True, progress_callback=None):
        seen["callback"] = progress_callback
        seen["do_eval"] = do_eval
        seen["max_concurrent"] = max_concurrent
        if False:
            yield {}

    monkeypatch.setattr(PaperVizProcessor, "process_queries_batch", fake_batch)
    callback = object()
    results = asyncio.run(
        gradio_app.process_parallel_candidates(
            [{"content": "method"}],
            progress_callback=callback,
        )
    )

    assert results == []
    assert seen["callback"] is callback
    assert seen["do_eval"] is False
    assert seen["max_concurrent"] == 10


def test_batch_without_progress_callback_keeps_existing_results(tmp_path):
    class RevisingCritic:
        async def process(self, data, source="stylist"):
            round_idx = data["current_critic_round"]
            data[f"target_diagram_critic_suggestions{round_idx}"] = "Keep revising."
            data[f"target_diagram_critic_desc{round_idx}"] = "revised"
            return data

    processor = _processor(
        _config(tmp_path, exp_mode="demo_planner_critic", max_critic_rounds=1),
        RevisingCritic(),
    )

    async def collect():
        results = []
        async for item in processor.process_queries_batch(
            [{"content": "method", "max_critic_rounds": 1}],
            do_eval=False,
        ):
            results.append(item)
        return results

    results = asyncio.run(collect())
    assert results[0]["eval_image_field"] == "target_diagram_critic_desc0_base64_jpg"
    assert "top10_references" in results[0]
