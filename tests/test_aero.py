import numpy as np
import pytest

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.airframe import UAV


@pytest.fixture
def aero():
    return AeroModel()


def test_wing_geometry():
    w = UAV.wing
    assert w.aspect_ratio == pytest.approx(10.0**2 / 30.0)
    assert w.sweep_deg(0.0) == pytest.approx(w.sweep_le_deg)
    assert w.sweep_deg(0.25) < w.sweep_le_deg


def test_lift_slope_below_thin_wing_theory(aero):
    assert 0 < aero.cl_alpha(0.0) < 2 * np.pi


def test_lift_slope_rises_with_mach(aero):
    assert aero.cl_alpha(0.85) > aero.cl_alpha(0.6) > aero.cl_alpha(0.0)


def test_statically_stable(aero):
    for M in (0.0, 0.6, 0.85):
        assert aero.cm_alpha(M) < 0


def test_no_wave_drag_at_low_mach(aero):
    assert aero.cd_wave(0.3, 0.5) == 0.0


def test_wave_drag_grows_past_drag_divergence(aero):
    cl = 0.2
    m_dd = aero.mach_dd(cl)
    assert aero.cd_wave(cl, m_dd + 0.05) > aero.cd_wave(cl, m_dd)


def test_max_lift_to_drag_is_reasonable(aero):
    ld_max = 0.5 / np.sqrt(aero.cd0 * aero.k_induced)
    assert 8 < ld_max < 16