import asyncio
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import demo


def _write_json(directory: Path, name: str, payload) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@contextmanager
def _patched_results_dir():
    """Point the loader's allowed root at a temporary results/demo directory."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "results" / "demo"
        root.mkdir(parents=True)
        with mock.patch.object(demo, "demo_results_directory", return_value=root):
            yield root


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
        with _patched_results_dir() as root:
            path = _write_json(root, "demo_20261018_143000.json", results)
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
        with _patched_results_dir() as root:
            path = _write_json(root, "demo_20260102_030405.json", payload)
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
        with _patched_results_dir() as root:
            path = _write_json(root, "demo_not-a-date.json", payload)
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
        with _patched_results_dir() as root:
            for index, payload in enumerate(payloads):
                path = _write_json(root, f"demo_bad_{index}.json", payload)
                with self.assertRaises(ValueError):
                    demo.load_saved_results(path)

    def test_invalid_json_text_raises(self) -> None:
        with _patched_results_dir() as root:
            path = root / "demo_invalid.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                demo.load_saved_results(path)
        self.assertIn("not valid", str(caught.exception))

    def test_non_dict_entries_raise(self) -> None:
        payloads = [
            ["not-an-object"],
            [{"content": "ok"}, "later"],
            {"results": [None]},
        ]
        with _patched_results_dir() as root:
            for index, payload in enumerate(payloads):
                path = _write_json(root, f"demo_nondict_{index}.json", payload)
                with self.assertRaises(ValueError) as caught:
                    demo.load_saved_results(path)
                self.assertIn("list of objects", str(caught.exception))

    def test_symlink_outside_results_demo_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "results" / "demo"
            root.mkdir(parents=True)
            outside = Path(tmp) / "secret.json"
            outside.write_text(json.dumps([{"content": "secret", "caption": "hidden"}]), encoding="utf-8")
            link = root / "demo_escape.json"
            link.symlink_to(outside)
            with mock.patch.object(demo, "demo_results_directory", return_value=root):
                with self.assertRaises(ValueError) as caught:
                    demo.load_saved_results(link)
            self.assertIn("results/demo", str(caught.exception))
            self.assertFalse(link.resolve().is_relative_to(root.resolve()))

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

            asyncio.run(demo.process_parallel_candidates([{}, {}], max_concurrent=0))
            self.assertEqual(captured["max_concurrent"], 1)
            asyncio.run(demo.process_parallel_candidates([{}], max_concurrent=-3))
            self.assertEqual(captured["max_concurrent"], 1)

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


class DemoLoadRestoresInputsTests(unittest.TestCase):
    def test_load_clears_example_selectors_and_shows_saved_text(self) -> None:
        from streamlit.testing.v1 import AppTest

        results_dir = demo.demo_results_directory()
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / "demo_20261018_150000.json"
        path.write_text(
            json.dumps([{"content": "Restored method.", "caption": "Restored caption."}]),
            encoding="utf-8",
        )
        try:
            at = AppTest.from_file(str(Path(demo.__file__).resolve()), default_timeout=30)
            at.run(timeout=30)
            at.selectbox(key="method_example_selector").set_value("PaperBanana Framework").run(timeout=30)
            at.selectbox(key="caption_example_selector").set_value("PaperBanana Framework").run(timeout=30)
            at.sidebar.selectbox(key="tab1_saved_result_file").set_value(
                "demo_20261018_150000.json (2026-10-18 15:00:00)"
            ).run(timeout=30)
            at.sidebar.button(key="tab1_load_saved_results").click().run(timeout=30)

            self.assertEqual(list(at.exception), [])
            self.assertEqual(at.selectbox(key="method_example_selector").value, "None")
            self.assertEqual(at.selectbox(key="caption_example_selector").value, "None")
            method_area = next(area for area in at.text_area if area.label.startswith("Method Section"))
            caption_area = next(area for area in at.text_area if area.label.startswith("Figure Caption"))
            self.assertEqual(method_area.value, "Restored method.")
            self.assertEqual(caption_area.value, "Restored caption.")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
