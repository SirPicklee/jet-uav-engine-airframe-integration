import numpy as np
import pytest

from jet_uav_sim.dynamics import LongitudinalModel, state_from_trim
from jet_uav_sim.trim import trim_level


@pytest.fixture(scope="module")
def model():
    return LongitudinalModel()


@pytest.mark.parametrize("altitude, mach", [(0.0, 0.3), (6000.0, 0.6), (12000.0, 0.8)])
def test_trim_is_an_equilibrium(model, altitude, mach):
    t = trim_level(altitude, mach, model.mass(1800.0), aero=model.aero, deck=model.deck)
    s = state_from_trim(t, 1800.0)
    d = model.derivatives(s, np.radians(t.de_deg), t.throttle)
    du, dw, dq, dtheta, dx, dh, dfuel, dthr = d
    assert abs(du) < 1e-6 and abs(dw) < 1e-6 and abs(dq) < 1e-6
    assert abs(dh) < 1e-6
    assert dx > 0 and dfuel < 0


def test_no_thrust_without_fuel(model):
    t = trim_level(6000.0, 0.6, model.mass(1800.0), aero=model.aero, deck=model.deck)
    s = state_from_trim(t, 0.0)
    d = model.derivatives(s, np.radians(t.de_deg), t.throttle)
    assert d[6] == 0.0      # no fuel burn
    assert d[0] < 0.0       # decelerating