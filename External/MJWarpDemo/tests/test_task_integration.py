import os
from pathlib import Path

import numpy as np
import pytest

from mjwarp_demo.task import PlanarPushTask


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("MJWARP_RUN_INTEGRATION") != "1", reason="set MJWARP_RUN_INTEGRATION=1")
def test_reset_is_deterministic_and_task_steps() -> None:
    model = Path(__file__).resolve().parents[1] / "model" / "planar_push.xml"
    task = PlanarPushTask(model, nworld=1)
    first = task.reset(123, "expert")
    task.step()
    second = task.reset(123, "expert")
    np.testing.assert_allclose(first["qpos"], second["qpos"], atol=1e-6)
    state = task.step()
    assert state["frame_id"] == 1
    assert len(state["action"]) == 2
