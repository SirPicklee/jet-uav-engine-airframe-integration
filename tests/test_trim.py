import pytest

from jet_uav_sim.trim import trim_level

CASES = [(0.0, 0.3), (6000.0, 0.6), (12000.0, 0.85)]


@pytest.mark.parametrize("altitude, mach", CASES)
def test_trim_converges_with_forces_balanced(altitude, mach):
    t = trim_level(altitude, mach, 6000.0)
    assert t.converged
    assert 0.0 < t.throttle < 1.0
    assert t.thrust == pytest.approx(t.drag, rel=0.01)


def test_specific_range_improves_with_altitude():
    low = trim_level(0.0, 0.6, 6000.0)
    high = trim_level(12000.0, 0.85, 6000.0)
    from jet_uav_sim.atmosphere import isa
    v_low = 0.6 * float(isa(0.0)[3])
    v_high = 0.85 * float(isa(12000.0)[3])
    assert v_high / high.fuel_flow > 1.5 * v_low / low.fuel_flow


def test_heavier_aircraft_needs_more_alpha():
    light = trim_level(6000.0, 0.6, 4500.0)
    heavy = trim_level(6000.0, 0.6, 6000.0)
    assert heavy.alpha_deg > light.alpha_deg