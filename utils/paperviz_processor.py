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
Processing pipeline of PaperVizAgent
"""

import asyncio
import time
from collections import Counter
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional

import numpy as np
from tqdm.asyncio import tqdm

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent

from .config import ExpConfig
from .eval_toolkits import get_score_for_image_referenced


def _is_placeholder_candidate_id(candidate_id: Any) -> bool:
    """Negative ids are sentinels and must not vote in stage aggregation."""
    return (
        isinstance(candidate_id, int)
        and not isinstance(candidate_id, bool)
        and candidate_id < 0
    )


def pipeline_stages(exp_mode: str, max_critic_rounds: int = 3) -> List[str]:
    """Ordered progress stages for one candidate.

    Adapted from #56 by @noking-xdj.
    """
    if exp_mode in ("dev_full", "demo_full", "dev_planner_stylist"):
        stages = ["Retriever", "Planner", "Stylist", "Visualizer"]
    elif exp_mode in ("dev_planner_critic", "demo_planner_critic", "dev_planner"):
        stages = ["Retriever", "Planner", "Visualizer"]
    else:
        return ["Processing"]

    try:
        rounds = max(0, int(max_critic_rounds or 0))
    except (TypeError, ValueError):
        rounds = 0
    if "critic" in exp_mode or "full" in exp_mode:
        for round_idx in range(rounds):
            stages.append(f"Critic R{round_idx}")
            stages.append(f"Visualizer (R{round_idx})")
    return stages


def planned_critic_rounds(data: Dict[str, Any], default_rounds: int) -> int:
    """Critic-round budget for one candidate, falling back to the config default."""
    if isinstance(data, dict) and "max_critic_rounds" in data:
        raw = data.get("max_critic_rounds")
    else:
        raw = default_rounds
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        try:
            return max(0, int(default_rounds or 0))
        except (TypeError, ValueError):
            return 0


def generation_progress_update(status: Dict[str, Any]) -> tuple:
    """Map tracker status onto the Gradio generation band between prepare and save."""
    pct = float(status.get("overall_pct") or 0)
    stage = status.get("stage") or "Working"
    done = status.get("stage_done", 0)
    total = status.get("total", 0)
    model = status.get("model") or ""
    desc = f"{stage} ({done}/{total})"
    if model:
        desc += f" [{model}]"
    return min(0.1 + pct * 0.8, 0.9), desc


class ProgressTracker:
    """Track pipeline progress across parallel candidates.

    Adapted from #56 by @noking-xdj.

    Each candidate has its own stage list, sized from that candidate's critic
    rounds. Overall progress is completed stages divided by the sum of those
    lists, so one candidate's shorter schedule cannot fill the bar while
    another candidate still has rounds left. Negative candidate ids are ignored
    and do not affect the active-stage vote or completion counts.
    """

    def __init__(
        self,
        candidate_stages: Dict[Any, List[str]],
        callback: Optional[Callable[..., Any]] = None,
    ):
        self._expected: Dict[Any, List[str]] = {}
        for candidate_id, stages in candidate_stages.items():
            if _is_placeholder_candidate_id(candidate_id):
                continue
            self._expected[candidate_id] = list(stages)

        self._completed: Dict[Any, set] = {cid: set() for cid in self._expected}
        self._candidate_stages: Dict[Any, str] = {}
        self._stage_done: Dict[str, int] = {}
        ordered: List[str] = []
        seen = set()
        for stages in self._expected.values():
            for stage in stages:
                if stage not in seen:
                    seen.add(stage)
                    ordered.append(stage)
                    self._stage_done[stage] = 0
        self.stages = ordered
        self._current_stage_start = time.monotonic()
        self._active_stage = ordered[0] if ordered else ""
        self._active_model = ""
        self._callback = callback

    @classmethod
    def from_batch(
        cls,
        exp_mode: str,
        data_list: List[Dict[str, Any]],
        default_rounds: int,
        callback: Optional[Callable[..., Any]] = None,
    ) -> "ProgressTracker":
        """Build a tracker from every candidate's own critic-round budget.

        The batch index is the tracking id. A payload ``candidate_id`` of -1
        is not used, so a mock-mode sentinel cannot join the stage vote.
        """
        candidate_stages: Dict[int, List[str]] = {}
        for index, data in enumerate(data_list):
            rounds = planned_critic_rounds(data, default_rounds)
            candidate_stages[index] = pipeline_stages(exp_mode, rounds)
        return cls(candidate_stages, callback=callback)

    @property
    def expected_stages(self) -> Dict[Any, List[str]]:
        return {cid: list(stages) for cid, stages in self._expected.items()}

    def enter_stage(self, candidate_id: Any, stage: str, model_name: str = "") -> None:
        """Record that a candidate has entered ``stage``."""
        if _is_placeholder_candidate_id(candidate_id) or candidate_id not in self._expected:
            return
        self._candidate_stages[candidate_id] = stage
        if model_name:
            self._active_model = model_name
        if self._candidate_stages:
            chosen = self._most_common_stage()
            if chosen != self._active_stage:
                self._active_stage = chosen
                self._current_stage_start = time.monotonic()
        self._notify()

    def _stage_rank(self, stage: str) -> int:
        try:
            return self.stages.index(stage)
        except ValueError:
            return -1

    def _most_common_stage(self) -> str:
        """Most common current stage. Ties resolve to the later pipeline stage.

        Equal counts would otherwise stick on whichever stage was entered
        first, hiding a candidate that is already on a later critic round.
        """
        counts = Counter(self._candidate_stages.values())
        top_count = max(counts.values())
        tied = [stage for stage, count in counts.items() if count == top_count]
        return max(tied, key=self._stage_rank)

    def complete_stage(self, candidate_id: Any, stage: str) -> None:
        """Record that a candidate finished ``stage``. Repeated calls are ignored."""
        if _is_placeholder_candidate_id(candidate_id) or candidate_id not in self._expected:
            return
        if stage not in self._expected[candidate_id]:
            return
        if stage in self._completed[candidate_id]:
            return
        self._completed[candidate_id].add(stage)
        self._stage_done[stage] = self._stage_done.get(stage, 0) + 1
        self._notify()

    def get_status(self) -> Dict[str, Any]:
        """Status dict consumed by the Gradio progress callback."""
        elapsed = time.monotonic() - self._current_stage_start
        completed_steps = sum(len(done) for done in self._completed.values())
        total_steps = sum(len(stages) for stages in self._expected.values())
        overall_pct = completed_steps / total_steps if total_steps else 0.0
        expected_for_stage = sum(
            1 for stages in self._expected.values() if self._active_stage in stages
        )
        return {
            "stage": self._active_stage,
            "model": self._active_model,
            "elapsed": int(elapsed),
            "stage_done": self._stage_done.get(self._active_stage, 0),
            "total": expected_for_stage,
            "overall_pct": min(overall_pct, 1.0),
            "all_stage_done": dict(self._stage_done),
        }

    def _notify(self) -> None:
        if self._callback:
            self._callback(self.get_status())


class PaperVizProcessor:
    """Main class for multimodal document processor"""

    def __init__(
        self,
        exp_config: ExpConfig,
        vanilla_agent: VanillaAgent,
        planner_agent: PlannerAgent,
        visualizer_agent: VisualizerAgent,
        stylist_agent: StylistAgent,
        critic_agent: CriticAgent,
        retriever_agent: RetrieverAgent,
        polish_agent: PolishAgent,
    ):
        self.exp_config = exp_config
        self.vanilla_agent = vanilla_agent
        self.planner_agent = planner_agent
        self.visualizer_agent = visualizer_agent
        self.stylist_agent = stylist_agent
        self.critic_agent = critic_agent
        self.retriever_agent = retriever_agent
        self.polish_agent = polish_agent

    async def _run_critic_iterations(
        self,
        data: Dict[str, Any],
        task_name: str,
        max_rounds: int = 3,
        source: str = "stylist",
        progress_tracker: Optional[ProgressTracker] = None,
        progress_candidate_id: Any = None,
    ) -> Dict[str, Any]:
        """
        Run multi-round critic iteration (up to max_rounds).
        Returns the data with critic suggestions and updated eval_image_field.
        
        Args:
            data: Input data dictionary
            task_name: Name of the task (e.g., "diagram", "plot")
            max_rounds: Maximum number of critic iterations
            source: Source of the input for round 0 critique ("stylist" or "planner")
        """
        # Determine initial fallback image key based on source
        if source == "planner":
            current_best_image_key = f"target_{task_name}_desc0_base64_jpg"
        else: # default to stylist
            current_best_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"

        max_rounds = max(0, int(max_rounds or 0))
        if max_rounds == 0:
            data["eval_image_field"] = current_best_image_key
            return data

        if progress_candidate_id is not None:
            cid = progress_candidate_id
        else:
            cid = data.get("candidate_id", 0)
            if _is_placeholder_candidate_id(cid):
                cid = 0

        def _enter(stage: str, model_name: str = "") -> None:
            if progress_tracker is not None:
                progress_tracker.enter_stage(cid, stage, model_name)

        def _done(stage: str) -> None:
            if progress_tracker is not None:
                progress_tracker.complete_stage(cid, stage)

        # Last round whose critic stage actually ran. Later planned rounds are
        # skipped on early stop and counted finished so this candidate's budget
        # does not stall the shared bar.
        completed_through = -1
        for round_idx in range(max_rounds):
            _enter(f"Critic R{round_idx}", self.exp_config.main_model_name)
            data["current_critic_round"] = round_idx
            data = await self.critic_agent.process(data, source=source)
            _done(f"Critic R{round_idx}")
            
            critic_suggestions_key = f"target_{task_name}_critic_suggestions{round_idx}"
            critic_suggestions = data.get(critic_suggestions_key, "")
            # Critic output occasionally parses to NaN (float) or None when the
            # model returns empty/malformed JSON on dense content. Coerce to str
            # before calling .strip() to avoid AttributeError mid-run.
            if not isinstance(critic_suggestions, str):
                critic_suggestions = "" if critic_suggestions is None or (
                    isinstance(critic_suggestions, float) and str(critic_suggestions) == "nan"
                ) else str(critic_suggestions)

            if critic_suggestions.strip() == "No changes needed.":
                print(f"[Critic Round {round_idx}] No changes needed. Stopping iteration.")
                _done(f"Visualizer (R{round_idx})")
                completed_through = round_idx
                break
            
            _enter(f"Visualizer (R{round_idx})", self.exp_config.image_gen_model_name)
            data = await self.visualizer_agent.process(data)
            _done(f"Visualizer (R{round_idx})")
            
            # Check if visualization validation succeeded
            new_image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
            if new_image_key in data and data[new_image_key]:
                current_best_image_key = new_image_key
                print(f"[Critic Round {round_idx}] Completed iteration. Visualization SUCCESS.")
                completed_through = round_idx
            else:
                print(f"[Critic Round {round_idx}] Visualization FAILED (No valid image). Rolling back to previous best: {current_best_image_key}")
                completed_through = round_idx
                break

        for remaining in range(completed_through + 1, max_rounds):
            _done(f"Critic R{remaining}")
            _done(f"Visualizer (R{remaining})")
        
        data["eval_image_field"] = current_best_image_key
        return data

    async def process_single_query(
        self,
        data: Dict[str, Any],
        do_eval=True,
        progress_tracker: Optional[ProgressTracker] = None,
        progress_candidate_id: Any = None,
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single query
        """
        # print(f"[DEBUG] -> Entered process_single_query for candidate {data.get('candidate_id', 'N/A')}")
        exp_mode = self.exp_config.exp_mode
        task_name = self.exp_config.task_name.lower()
        retrieval_setting = self.exp_config.retrieval_setting

        # Skip retriever if results were already populated by process_queries_batch
        already_retrieved = "top10_references" in data
        if progress_candidate_id is not None:
            cid = progress_candidate_id
        else:
            cid = data.get("candidate_id", 0)
            if _is_placeholder_candidate_id(cid):
                cid = 0

        def _enter(stage: str, model_name: str = "") -> None:
            if progress_tracker is not None:
                progress_tracker.enter_stage(cid, stage, model_name)

        def _done(stage: str) -> None:
            if progress_tracker is not None:
                progress_tracker.complete_stage(cid, stage)

        if exp_mode == "vanilla":
            _enter("Processing")
            data = await self.vanilla_agent.process(data)
            _done("Processing")
            data["eval_image_field"] = f"vanilla_{task_name}_base64_jpg"

        elif exp_mode == "dev_planner":
            if not already_retrieved:
                _enter("Retriever", self.exp_config.main_model_name)
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                _done("Retriever")
            _enter("Planner", self.exp_config.main_model_name)
            data = await self.planner_agent.process(data)
            _done("Planner")
            _enter("Visualizer", self.exp_config.image_gen_model_name)
            data = await self.visualizer_agent.process(data)
            _done("Visualizer")
            data["eval_image_field"] = f"target_{task_name}_desc0_base64_jpg"

        elif exp_mode == "dev_planner_stylist":
            if not already_retrieved:
                _enter("Retriever", self.exp_config.main_model_name)
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                _done("Retriever")
            _enter("Planner", self.exp_config.main_model_name)
            data = await self.planner_agent.process(data)
            _done("Planner")
            _enter("Stylist", self.exp_config.main_model_name)
            data = await self.stylist_agent.process(data)
            _done("Stylist")
            _enter("Visualizer", self.exp_config.image_gen_model_name)
            data = await self.visualizer_agent.process(data)
            _done("Visualizer")
            data["eval_image_field"] = f"target_{task_name}_stylist_desc0_base64_jpg"

        elif exp_mode in ["dev_planner_critic", "demo_planner_critic"]:
            if not already_retrieved:
                _enter("Retriever", self.exp_config.main_model_name)
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                _done("Retriever")
            _enter("Planner", self.exp_config.main_model_name)
            data = await self.planner_agent.process(data)
            _done("Planner")
            _enter("Visualizer", self.exp_config.image_gen_model_name)
            data = await self.visualizer_agent.process(data)
            _done("Visualizer")
            # Use max_critic_rounds from data (if set) or config
            max_rounds = data.get("max_critic_rounds", self.exp_config.max_critic_rounds)
            data = await self._run_critic_iterations(
                data,
                task_name,
                max_rounds=max_rounds,
                source="planner",
                progress_tracker=progress_tracker,
                progress_candidate_id=cid,
            )
            if "demo" in exp_mode: do_eval = False

        elif exp_mode in ["dev_full", "demo_full"]:
            if not already_retrieved:
                _enter("Retriever", self.exp_config.main_model_name)
                data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
                _done("Retriever")
            _enter("Planner", self.exp_config.main_model_name)
            data = await self.planner_agent.process(data)
            _done("Planner")
            _enter("Stylist", self.exp_config.main_model_name)
            data = await self.stylist_agent.process(data)
            _done("Stylist")
            _enter("Visualizer", self.exp_config.image_gen_model_name)
            data = await self.visualizer_agent.process(data)
            _done("Visualizer")
            # Use max_critic_rounds from data (if set) or config
            max_rounds = data.get("max_critic_rounds", self.exp_config.max_critic_rounds)
            data = await self._run_critic_iterations(
                data,
                task_name,
                max_rounds=max_rounds,
                source="stylist",
                progress_tracker=progress_tracker,
                progress_candidate_id=cid,
            )
            if "demo" in exp_mode: do_eval = False
        
        elif exp_mode == "dev_polish":
            _enter("Processing")
            data = await self.polish_agent.process(data)
            _done("Processing")
            data["eval_image_field"] = f"polished_{task_name}_base64_jpg"
        
        elif exp_mode == "dev_retriever":
            _enter("Processing")
            data = await self.retriever_agent.process(data)
            _done("Processing")
            do_eval = False

        else:
            raise ValueError(f"Unknown experiment name: {exp_mode}")

        if do_eval:
            data_with_eval = await self.evaluation_function(data, exp_config=self.exp_config)
            return data_with_eval
        else:
            return data

    async def process_queries_batch(
        self,
        data_list: List[Dict[str, Any]],
        max_concurrent: int = 50,
        do_eval: bool = True,
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Batch process queries with concurrency support.
        Retriever is run once before parallelization to avoid redundant API calls.
        """
        # Run Retriever once and share results across all candidates
        exp_mode = self.exp_config.exp_mode
        retrieval_setting = self.exp_config.retrieval_setting
        needs_retrieval = exp_mode not in ("vanilla", "dev_polish", "dev_retriever")
        # Stage lists follow each candidate's own max_critic_rounds. A failure
        # in one candidate still propagates to the caller and does not cancel
        # the other tasks.
        tracker = None
        if progress_callback is not None:
            tracker = ProgressTracker.from_batch(
                exp_mode,
                data_list,
                self.exp_config.max_critic_rounds,
                callback=progress_callback,
            )

        if needs_retrieval and data_list:
            if tracker is not None:
                for index in range(len(data_list)):
                    tracker.enter_stage(index, "Retriever", self.exp_config.main_model_name)
            print("[Retriever] Running retrieval once for all candidates...")
            first_data = data_list[0]
            first_data = await self.retriever_agent.process(first_data, retrieval_setting=retrieval_setting)
            retrieval_keys = ("top10_references", "retrieved_examples")
            for data in data_list[1:]:
                for key in retrieval_keys:
                    if key in first_data:
                        data[key] = first_data[key]
            print(f"[Retriever] Done. Retrieved {len(first_data.get('top10_references', []))} references.")
            if tracker is not None:
                for index in range(len(data_list)):
                    tracker.complete_stage(index, "Retriever")

        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_with_semaphore(index, doc):
            async with semaphore:
                return await self.process_single_query(
                    doc,
                    do_eval=do_eval,
                    progress_tracker=tracker,
                    progress_candidate_id=index,
                )

        # Create all tasks
        tasks = []
        for index, data in enumerate(data_list):
            task = asyncio.create_task(process_with_semaphore(index, data))
            tasks.append(task)
        
        all_result_list = []
        eval_dims = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]

        with tqdm(total=len(tasks), desc="Processing concurrently",ascii=True) as pbar:
            # Iterate through completed tasks returned by as_completed
            for future in asyncio.as_completed(tasks):
                result_data = await future
                all_result_list.append(result_data)
                postfix_dict = {}

                for dim in eval_dims:
                    winner_key = f"{dim}_outcome"
                    if winner_key in result_data:
                        winners = [d.get(winner_key) for d in all_result_list]
                        total = len(winners)

                        if total > 0:
                            h_cnt = winners.count("Human")
                            m_cnt = winners.count("Model")
                            t_cnt = winners.count("Tie") + winners.count("Both are good") + winners.count("Both are bad")

                            h_rate = (h_cnt / total) * 100
                            m_rate = (m_cnt / total) * 100
                            t_rate = (t_cnt / total) * 100

                            display_key = dim[:5].capitalize()
                            postfix_dict[display_key] = f"{m_rate:.0f}/{t_rate:.0f}/{h_rate:.0f}"

                pbar.set_postfix(postfix_dict)
                pbar.update(1)
                yield result_data

    async def evaluation_function(
        self, data: Dict[str, Any], exp_config: ExpConfig
    ) -> Dict[str, Any]:
        """
        Evaluation function - uses referenced setting (GT shown first)
        """
        data = await get_score_for_image_referenced(
            data, task_name=exp_config.task_name, work_dir=exp_config.work_dir
        )
        return data

