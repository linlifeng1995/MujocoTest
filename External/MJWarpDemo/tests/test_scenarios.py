from pathlib import Path

import mujoco
import pytest

from mjwarp_demo.scenarios import SCENARIOS, get_scenario, scenario_summaries


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_scenario_models_compile_and_expose_two_actions(scenario_id: str) -> None:
    definition = get_scenario(scenario_id)
    model = mujoco.MjModel.from_xml_path(str(definition.model_path(PACKAGE_ROOT)))
    assert model.nu == 2
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.ngeom > 0


def test_scenario_catalog_is_complete() -> None:
    summaries = scenario_summaries()
    assert {item["scenario_id"] for item in summaries} == set(SCENARIOS)
    assert all(item["display_name"] and item["business_type"] for item in summaries)
