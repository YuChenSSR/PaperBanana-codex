"""Dataset acquisition for the headless skill CLI.

``ensure_dataset`` used to download ``PaperBananaBench.zip`` and rename
CP437-mojibake image files. The Hub dataset now publishes Parquet, and
``prepare`` writes ``ref.json``, ``test.json``, and ``images/`` directly.
These tests pin that path: the short-circuit, a failed download, and an
offline hub fake that returns real Parquet. No test here opens a socket.
"""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from skill import run as skill_run


IMAGE_TYPE = pa.struct([
    pa.field("bytes", pa.binary()),
    pa.field("path", pa.string()),
])


def _parquet_row(task: str) -> dict:
    image = {"bytes": b"not-a-real-jpeg", "path": None}
    if task == "diagram":
        return {
            "id": "x",
            "content": "method text",
            "visual_intent": "show it",
            "path_to_gt_image": "images/x.jpg",
            "category": "method",
            "source_pdf": "paper.pdf",
            "diagram_page_num": 1,
            "method_section_title": "Method",
            "width": 100,
            "height": 80,
            "rounded_ratio": "1:1",
            "split": "test",
            "image": image,
        }
    return {
        "id": "x",
        "content": json.dumps({"x": [1]}),
        "visual_intent": "plot it",
        "path_to_gt_image": "images/x.jpg",
        "original_category": "line",
        "gt_code": "plt.plot([1])",
        "width": 100,
        "height": 80,
        "rounded_ratio": "1:1",
        "difficulty": "easy",
        "category": "line",
        "image": image,
    }


def _write_parquet(path: Path, row: dict) -> None:
    arrays = []
    names = []
    for key, value in row.items():
        names.append(key)
        if key == "image":
            arrays.append(pa.array([value], type=IMAGE_TYPE))
        else:
            arrays.append(pa.array([value]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


class FakeHub:
    """Stands in for ``huggingface_hub``; records calls, never touches a socket."""

    def __init__(self, *, side_effect=None, bad_schema=False) -> None:
        self.calls: list[dict] = []
        self.side_effect = side_effect
        self.bad_schema = bad_schema
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def close(self) -> None:
        self._tmp.cleanup()

    def hf_hub_download(self, repo_id, filename, **kwargs):
        self.calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        if self.side_effect is not None:
            raise self.side_effect

        path = self.root / filename
        if self.bad_schema:
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"only": [1]}), path)
            return str(path)

        task = Path(filename).parts[-2]
        _write_parquet(path, _parquet_row(task))
        return str(path)


@contextlib.contextmanager
def fake_hub(**kwargs):
    hub = FakeHub(**kwargs)
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hub.hf_hub_download
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = module
    try:
        yield hub
    finally:
        hub.close()
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved


class DatasetAcquisitionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        patcher = unittest.mock.patch.object(skill_run, "PROJECT_ROOT", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def acquire(self, task="diagram"):
        with contextlib.redirect_stdout(io.StringIO()):
            skill_run.ensure_dataset(task)

    def task_dir(self, task="diagram") -> Path:
        return self.tmp / "data" / "PaperBananaBench" / task

    def assert_downloaded(self, hub: FakeHub, task: str) -> None:
        self.assertEqual(
            [call["filename"] for call in hub.calls],
            [
                f"data/{task}/test-00000-of-00001.parquet",
                f"data/{task}/ref-00000-of-00001.parquet",
            ],
        )
        for call in hub.calls:
            self.assertEqual(call["repo_id"], "dwzhu/PaperBananaBench")
            self.assertEqual(call["repo_type"], "dataset")


class PrepareWiringTests(DatasetAcquisitionTestCase):
    def test_prepare_is_loaded_from_the_repo_script(self) -> None:
        skill_run._load_prepare.fn = None
        self.addCleanup(lambda: setattr(skill_run._load_prepare, "fn", None))

        loaded = skill_run._load_prepare()

        expected = (
            Path(skill_run.__file__).resolve().parent.parent
            / "scripts"
            / "prepare_bench.py"
        )
        self.assertEqual(Path(loaded.__code__.co_filename).resolve(), expected)

    def test_a_skill_install_without_the_script_uses_the_vendored_prepare(self) -> None:
        skill_run._load_prepare.fn = None
        self.addCleanup(lambda: setattr(skill_run._load_prepare, "fn", None))

        with unittest.mock.patch.object(
            skill_run, "_prepare_script_candidates", return_value=()
        ):
            loaded = skill_run._load_prepare()

        self.assertIs(loaded, skill_run._vendored_prepare)


class ExtractionTests(DatasetAcquisitionTestCase):
    def test_prepare_produces_the_layout_the_retriever_reads(self) -> None:
        with fake_hub() as hub:
            self.acquire()

        self.assert_downloaded(hub, "diagram")
        self.assertTrue((self.task_dir() / "ref.json").exists())
        self.assertTrue((self.task_dir() / "test.json").exists())
        self.assertTrue((self.task_dir() / "images").is_dir())

    def test_the_image_paths_in_the_json_resolve_where_the_planner_looks(self) -> None:
        """PlannerAgent opens <task>/<path_to_gt_image> with no guard."""
        with fake_hub():
            self.acquire()

        for filename in ("ref.json", "test.json"):
            entries = json.loads(
                (self.task_dir() / filename).read_text(encoding="utf-8")
            )
            for entry in entries:
                self.assertTrue((self.task_dir() / entry["path_to_gt_image"]).is_file())

    def test_a_second_run_makes_no_network_call(self) -> None:
        with fake_hub() as first:
            self.acquire()
        self.assertEqual(len(first.calls), 2)

        with fake_hub() as second:
            self.acquire()
        self.assertEqual(second.calls, [], "the presence check did not short-circuit")

    def test_an_unexpected_parquet_schema_raises_instead_of_half_landing(self) -> None:
        with fake_hub(bad_schema=True):
            with self.assertRaises(KeyError):
                self.acquire()

        self.assertFalse(
            task_dir_usable_via_module(self.task_dir()),
            "a failed acquisition left a tree that satisfies the presence check",
        )

    def test_a_failed_download_leaves_nothing_usable_behind(self) -> None:
        with fake_hub(side_effect=RuntimeError("network died")):
            with self.assertRaises(RuntimeError):
                self.acquire()

        self.assertFalse(task_dir_usable_via_module(self.task_dir()))

    def test_missing_dataset_dependencies_exit_with_an_install_hint(self) -> None:
        module = types.ModuleType("huggingface_hub")
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = module
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    self.acquire()
        finally:
            if saved is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = saved

        self.assertEqual(caught.exception.code, 1)
        message = stderr.getvalue()
        self.assertIn("huggingface_hub", message)
        self.assertIn("pyarrow", message)
        self.assertIn("pip install", message)
        self.assertFalse(task_dir_usable_via_module(self.task_dir()))


class IncompleteTaskDirectoryTests(DatasetAcquisitionTestCase):
    """A half-populated task directory must be repaired, not stranded."""

    def partial(self, *, ref: bool, images: bool) -> Path:
        task_dir = self.task_dir()
        task_dir.mkdir(parents=True)
        if ref:
            (task_dir / "ref.json").write_text("[]", encoding="utf-8")
        if images:
            (task_dir / "images").mkdir()
        return task_dir

    def test_a_task_dir_missing_images_is_repaired(self) -> None:
        self.partial(ref=True, images=False)

        with fake_hub():
            self.acquire()

        self.assertTrue(task_dir_usable_via_module(self.task_dir()))

    def test_an_empty_task_dir_is_repaired(self) -> None:
        self.partial(ref=False, images=False)

        with fake_hub():
            self.acquire()

        self.assertTrue(task_dir_usable_via_module(self.task_dir()))

    def test_repairing_stops_the_endless_re_download(self) -> None:
        self.partial(ref=True, images=False)

        with fake_hub() as first:
            self.acquire()
        self.assertEqual(len(first.calls), 2)

        with fake_hub() as again:
            self.acquire()
        self.assertEqual(again.calls, [], "a repaired dataset downloaded a second time")

    def test_an_already_complete_task_dir_is_left_alone(self) -> None:
        task_dir = self.partial(ref=False, images=True)
        (task_dir / "ref.json").write_text('["LOCAL"]', encoding="utf-8")

        with fake_hub() as hub:
            self.acquire()

        self.assertEqual(hub.calls, [], "a complete dataset triggered a download")
        self.assertIn("LOCAL", (task_dir / "ref.json").read_text(encoding="utf-8"))

    def test_a_sibling_task_is_not_disturbed_while_repairing_another(self) -> None:
        base = self.tmp / "data" / "PaperBananaBench"
        (base / "plot" / "images").mkdir(parents=True)
        (base / "plot" / "ref.json").write_text('["PLOT-LOCAL"]', encoding="utf-8")
        self.partial(ref=True, images=False)

        with fake_hub() as hub:
            self.acquire()

        self.assertTrue(task_dir_usable_via_module(self.task_dir()))
        self.assert_downloaded(hub, "diagram")
        self.assertIn(
            "PLOT-LOCAL", (base / "plot" / "ref.json").read_text(encoding="utf-8")
        )


def task_dir_usable_via_module(task_dir: Path) -> bool:
    """Assert through the module's own predicate, not a copy of it."""
    return skill_run.task_dir_usable(task_dir)


if __name__ == "__main__":
    unittest.main()
