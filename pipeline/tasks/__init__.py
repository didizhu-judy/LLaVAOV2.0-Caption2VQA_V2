from __future__ import annotations

from pipeline.tasks.base import TaskPlugin
from pipeline.tasks.benchmark_similarity import BenchmarkSimilarityTask
from pipeline.tasks.caption_to_vqa import CaptionToVQATask
from pipeline.tasks.clean_mm_qa import CleanMMQATask

_TASKS: dict[str, type[TaskPlugin]] = {
    BenchmarkSimilarityTask.name: BenchmarkSimilarityTask,
    CaptionToVQATask.name: CaptionToVQATask,
    CleanMMQATask.name: CleanMMQATask,
}


def get_task_plugin(task_name: str) -> TaskPlugin:
    task_key = (task_name or "").strip()
    if task_key not in _TASKS:
        supported = ", ".join(sorted(_TASKS))
        raise ValueError(f"Unknown task_name '{task_name}'. Supported tasks: {supported}")
    return _TASKS[task_key]()


def list_task_names() -> list[str]:
    return sorted(_TASKS.keys())
