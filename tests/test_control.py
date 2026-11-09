import numpy as np
import pytest

from jet_uav_sim.atmosphere import isa
from jet_uav_sim.control import PI
from jet_uav_sim.sim import simulate_autopilot


def test_pi_anti_windup_stops_integration_when_saturated():
    pi = PI(kp=1.0, ki=1.0, out_min=0.0, out_max=1.0, feedforward=0.5)
    for _ in range(100):
        pi.update(10.0, 0.1)            # large error, output stuck at 1.0
    assert pi.integral == 0.0


@pytest.fixture(scope="module")
def altitude_step():
    alt, mach = 6000.0, 0.6
    V0 = mach * float(isa(alt)[3])
    return simulate_autopilot(alt, mach, 1800.0,
                              V_cmd_fn=lambda t: V0,
                              h_cmd_fn=lambda t: alt + (200.0 if t >= 5.0 else 0.0),
                              t_end=200.0), alt, V0


def test_altitude_step_reaches_target(altitude_step):
    r, alt, V0 = altitude_step
    assert abs(r["h"][-1] - (alt + 200.0)) < 2.0
    assert r["h"].max() < alt + 200.0 * 1.10          # overshoot below 10 %


def test_speed_held_during_climb(altitude_step):
    r, alt, V0 = altitude_step
    assert abs(r.V[-1] - V0) < 0.5


def test_elevator_stays_gentle(altitude_step):
    r, _, _ = altitude_step
    assert np.abs(np.degrees(r.de)).max() < 5.0