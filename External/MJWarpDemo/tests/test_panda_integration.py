from pathlib import Path

import pytest

from mjwarp_demo.task import create_task


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize("scenario_id", ["panda_pick_place", "panda_peg_insert"])
def test_panda_expert_seed_zero_reaches_success(scenario_id: str) -> None:
    task = create_task(scenario_id, PACKAGE_ROOT, device="cuda:0")
    for _ in range(task.definition.max_frames):
        state = task.step()
        if state["terminated"]:
            break
    assert state["success"], state["termination_reason"]
    assert state["termination_reason"] == "success"
