"""Task definitions for robot manipulation benchmarks."""
from envs.task_definitions.pick_and_place import PickAndPlaceTask
from envs.task_definitions.sort_objects import SortObjectsTask
from envs.task_definitions.stack_blocks import StackBlocksTask

TASK_REGISTRY: dict = {
    "pick_and_place": PickAndPlaceTask,
    "sort_objects": SortObjectsTask,
    "stack_blocks": StackBlocksTask,
}


def get_task(name: str, config: dict):
    """Instantiate a task by name."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name](config)


__all__ = [
    "PickAndPlaceTask",
    "SortObjectsTask",
    "StackBlocksTask",
    "TASK_REGISTRY",
    "get_task",
]
