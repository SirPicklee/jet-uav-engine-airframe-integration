import numpy as np
import pytest

from jet_uav_sim.airframe import UAV
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.mission import Mission
from jet_uav_sim.trim import trim_level


@pytest.fixture(scope="module")
def climbed():
    m = Mission(mass0=UAV.mass.mtow, fuel0=UAV.mass.fuel_max)
    m.climb(10000.0, mach=0.6)
    return m


def test_climb_reaches_altitude_and_burns_fuel(climbed):
    assert climbed.h == pytest.approx(10000.0)
    assert 0 < climbed.fuel_used < 200
    assert climbed.mass == pytest.approx(UAV.mass.mtow - climbed.fuel_used)


def test_cruise_matches_breguet(climbed):
    m = Mission(mass0=climbed.mass, fuel0=UAV.mass.fuel_max)
    m.h = climbed.h
    mach, dist = 0.75, 300e3
    t = trim_level(m.h, mach, m.mass, aero=m.aero, deck=m.deck)
    V = mach * float(isa(m.h)[3])
    c = t.fuel_flow / t.thrust
    expected = m.mass * (1 - np.exp(-dist * c * G0 / (V * t.lift_to_drag)))
    m.cruise(mach, distance_m=dist)
    assert m.fuel_used == pytest.approx(expected, rel=0.02)


def test_out_of_fuel_raises():
    m = Mission(mass0=UAV.mass.mtow, fuel0=50.0)
    with pytest.raises(RuntimeError, match="out of fuel"):
        m.climb(10000.0, mach=0.6)