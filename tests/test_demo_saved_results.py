import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import demo


def _write_json(directory: Path, name: str, payload) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class DemoSavedResultsLoaderTests(unittest.TestCase):
    def test_loads_current_list_format(self) -> None:
        results = [
            {
                "content": "method text",
                "caption": "figure caption",
                "target_diagram_stylist_desc0": "styled",
                "target_diagram_critic_desc1_base64_jpg": "img",
            }
        ]
        with TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp), "demo_20261018_143000.json", results)
            loaded = demo.load_saved_results(path)

        self.assertEqual(loaded["results"], results)
        self.assertEqual(loaded["exp_mode"], "demo_full")
        self.assertEqual(loaded["timestamp"], "2026-10-18 14:30:00")
        self.assertEqual(loaded["method_content"], "method text")
        self.assertEqual(loaded["caption"], "figure caption")
        self.assertEqual(Path(loaded["json_file"]), path.resolve())

    def test_loads_wrapped_metadata_dict_without_changing_list_fields(self) -> None:
        payload = {
            "metadata": {
                "timestamp": "2026-01-02 03:04:05",
                "exp_mode": "demo_planner_critic",
                "method_content": "from metadata",
                "caption": "caption from metadata",
            },
            "results": [
                {
                    "content": "from result",
                    "caption": "caption from result",
                    "target_diagram_stylist_desc0": "styled",
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp), "demo_20260102_030405.json", payload)
            loaded = demo.load_saved_results(path)

        self.assertEqual(loaded["results"], payload["results"])
        self.assertEqual(loaded["exp_mode"], "demo_planner_critic")
        self.assertEqual(loaded["timestamp"], "2026-01-02 03:04:05")
        self.assertEqual(loaded["method_content"], "from metadata")
        self.assertEqual(loaded["caption"], "caption from metadata")

    def test_wrapped_dict_falls_back_to_result_fields(self) -> None:
        payload = {
            "results": [
                {
                    "content": {"x": [1, 2]},
                    "caption": "plot caption",
                    "target_plot_critic_desc0_base64_jpg": "img",
                }
            ]
        }
        with TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp), "demo_not-a-date.json", payload)
            loaded = demo.load_saved_results(path)

        self.assertEqual(loaded["exp_mode"], "demo_planner_critic")
        self.assertEqual(loaded["timestamp"], "not-a-date")
        self.assertEqual(loaded["method_content"], json.dumps({"x": [1, 2]}, indent=2))
        self.assertEqual(loaded["caption"], "plot caption")

    def test_plot_stylist_results_infer_full_pipeline(self) -> None:
        mode = demo.infer_exp_mode_from_results(
            [{"target_plot_stylist_desc0_base64_jpg": "img"}]
        )
        self.assertEqual(mode, "demo_full")

    def test_empty_and_unrecognized_results_use_current_default_pipeline(self) -> None:
        self.assertEqual(demo.infer_exp_mode_from_results([]), "demo_full")
        self.assertEqual(demo.infer_exp_mode_from_results([{"content": "only text"}]), "demo_full")

    def test_invalid_payloads_raise(self) -> None:
        payloads = [
            "not-an-object",
            3,
            {"results": {"not": "a list"}},
            {"metadata": ["bad"], "results": []},
        ]
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for index, payload in enumerate(payloads):
                path = _write_json(directory, f"demo_bad_{index}.json", payload)
                with self.assertRaises(ValueError):
                    demo.load_saved_results(path)

    def test_lists_saved_demo_json_newest_first(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_json(directory, "demo_20260101_000000.json", [])
            _write_json(directory, "demo_20261018_120000.json", [])
            (directory / "notes.json").write_text("[]", encoding="utf-8")
            (directory / "other_demo.json").write_text("[]", encoding="utf-8")

            listed = demo.list_saved_result_files(directory)

        self.assertEqual(
            [path.name for path in listed],
            ["demo_20261018_120000.json", "demo_20260101_000000.json"],
        )

    def test_missing_results_directory_lists_nothing(self) -> None:
        missing = Path("/tmp/paperbanana-missing-demo-results")
        self.assertEqual(demo.list_saved_result_files(missing), [])

    def test_timestamp_from_filename(self) -> None:
        self.assertEqual(
            demo.infer_timestamp_from_filename(Path("demo_20261018_143000.json")),
            "2026-10-18 14:30:00",
        )
        self.assertEqual(demo.infer_timestamp_from_filename(Path("notes.json")), "notes")


class DemoConcurrencyTests(unittest.TestCase):
    def test_sidebar_value_is_forwarded_and_capped_by_batch_size(self) -> None:
        captured = {}

        class FakeProcessor:
            def __init__(self, exp_config, **_agents):
                captured["task_name"] = exp_config.task_name

            async def process_queries_batch(self, data_list, max_concurrent=50, do_eval=True):
                captured["max_concurrent"] = max_concurrent
                captured["do_eval"] = do_eval
                for item in data_list:
                    yield {"candidate_id": item.get("candidate_id")}

        with self._patched_pipeline(FakeProcessor):
            limited = asyncio.run(
                demo.process_parallel_candidates([{}, {}, {}], max_concurrent=2, task_name="diagram")
            )
            self.assertEqual(captured["max_concurrent"], 2)
            self.assertFalse(captured["do_eval"])
            self.assertEqual(len(limited), 3)

            asyncio.run(demo.process_parallel_candidates([{}, {}], max_concurrent=10))
            self.assertEqual(captured["max_concurrent"], 2)

            defaulted = asyncio.run(demo.process_parallel_candidates([{} for _ in range(12)]))
            self.assertEqual(captured["max_concurrent"], 10)
            self.assertEqual(len(defaulted), 12)
            self.assertEqual(captured["task_name"], "diagram")

    def _patched_pipeline(self, fake_processor):
        class DummyAgent:
            def __init__(self, **_kwargs):
                pass

        return mock.patch.multiple(
            demo,
            PaperVizProcessor=fake_processor,
            VanillaAgent=DummyAgent,
            PlannerAgent=DummyAgent,
            VisualizerAgent=DummyAgent,
            StylistAgent=DummyAgent,
            CriticAgent=DummyAgent,
            RetrieverAgent=DummyAgent,
            PolishAgent=DummyAgent,
        )


if __name__ == "__main__":
    unittest.main()
