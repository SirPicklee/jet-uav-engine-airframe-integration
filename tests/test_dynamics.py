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

def test_longitudinal_modes(model):
    from jet_uav_sim.linearize import linearize, modes
    t = trim_level(6000.0, 0.6, model.mass(1800.0), aero=model.aero, deck=model.deck)
    A, _ = linearize(model, state_from_trim(t, 1800.0), np.radians(t.de_deg), t.throttle)
    osc, real = modes(A)
    phugoid, short_period = osc
    assert 0.04 < phugoid["zeta"] < 0.2                  # MIL-F-8785C level 1
    assert 0.30 < short_period["zeta"] < 2.0             # level 1, category B
    assert short_period["wn"] > 10 * phugoid["wn"]       # modes well separated
    assert any(abs(r["time_constant"] - model.tau_engine) < 1e-3 for r in real)