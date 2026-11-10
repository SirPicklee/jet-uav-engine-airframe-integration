import numpy as np
import pytest

from jet_uav_sim.dynamics import LongitudinalModel, state_from_trim
from jet_uav_sim.dynamics6dof import SixDofModel, state6_from_trim
from jet_uav_sim.trim import trim_level

# where each 3-DoF state sits in the 6-DoF state vector
IDX_3_IN_6 = [0, 2, 4, 7, 9, 11, 12, 13]     # u, w, q, theta, x, h, fuel, throttle
LATERAL = [1, 3, 5, 6, 8, 10]                # v, p, r, phi, psi, y


@pytest.fixture(scope="module")
def models():
    m3 = LongitudinalModel()
    m6 = SixDofModel(aero=m3.aero, deck=m3.deck)
    t = trim_level(6000.0, 0.6, m3.mass(1800.0), aero=m3.aero, deck=m3.deck)
    return m3, m6, t


def test_reduces_to_3dof_away_from_trim(models):
    """With the lateral states at zero, the 6-DoF model is the 3-DoF model."""
    m3, m6, t = models
    s3 = state_from_trim(t, 1800.0) + np.array([5.0, 2.0, 0.05, 0.02, 0, 0, 0, 0.1])
    s6 = np.zeros(14)
    s6[IDX_3_IN_6] = s3
    de, thr = np.radians(t.de_deg) + 0.01, 0.7
    d3 = m3.derivatives(s3, de, thr)
    d6 = m6.derivatives(s6, 0.0, de, 0.0, thr)
    np.testing.assert_allclose(d6[IDX_3_IN_6], d3, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d6[LATERAL], 0.0, atol=1e-12)


def test_trim_is_an_equilibrium(models):
    _, m6, t = models
    d = m6.derivatives(state6_from_trim(t, 1800.0), 0.0, np.radians(t.de_deg), 0.0,
                       t.throttle)
    np.testing.assert_allclose(d[:9], 0.0, atol=1e-6)        # velocities, rates, angles
    assert abs(d[11]) < 1e-6                                 # altitude


def test_heading_east_moves_east(models):
    _, m6, t = models
    d = m6.derivatives(state6_from_trim(t, 1800.0, heading=np.pi / 2), 0.0,
                       np.radians(t.de_deg), 0.0, t.throttle)
    dx, dy = d[9], d[10]
    assert abs(dx) < 1e-9 and dy > 100.0


def test_control_signs(models):
    _, m6, t = models
    s = state6_from_trim(t, 1800.0)
    de, thr = np.radians(t.de_deg), t.throttle
    assert m6.derivatives(s, 0.05, de, 0.0, thr)[3] > 0      # +aileron: roll right
    assert m6.derivatives(s, 0.0, de, 0.05, thr)[5] < 0      # +rudder: nose left



def test_lateral_modes_meet_flying_qualities(models):
    from jet_uav_sim.linearize import linearize_lateral, modes
    _, m6, t = models
    A, _ = linearize_lateral(m6, state6_from_trim(t, 1800.0),
                             (0.0, np.radians(t.de_deg), 0.0, t.throttle))
    osc, real = modes(A)
    assert len(osc) == 1 and len(real) == 2           # Dutch roll + roll + spiral
    dutch = osc[0]
    roll = max(real, key=lambda m: abs(m["eigenvalue"]))
    spiral = min(real, key=lambda m: abs(m["eigenvalue"]))
    assert dutch["zeta"] >= 0.08 and dutch["zeta"] * dutch["wn"] >= 0.15
    assert roll["time_constant"] <= 1.4
    assert spiral["eigenvalue"] < 0                   # spiral stable
    assert m6.lateral.spiral_stable()                 # agrees with the derivative test