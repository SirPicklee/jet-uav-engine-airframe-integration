import numpy as np
import pytest

from jet_uav_sim.engine import make_engine
from jet_uav_sim.thrust_deck import ThrustDeck, generate


@pytest.fixture(scope="module")
def deck():
    df = generate(np.array([0.0, 10000.0]), np.array([0.0, 0.8]),
                  np.array([0.0, 1.0]), verbose=False)
    return ThrustDeck(df)


def test_design_point_thrust():
    design, _ = make_engine()
    assert 25e3 < design.net_thrust < 33e3


def test_max_point_matches_design(deck):
    design, _ = make_engine()
    assert deck.thrust(0.0, 0.0, 1.0) == pytest.approx(design.net_thrust, rel=0.01)


def test_thrust_lapse_with_altitude(deck):
    assert deck.thrust(10000.0, 0.8, 1.0) < 0.5 * deck.thrust(0.0, 0.0, 1.0)


def test_throttle_reduces_thrust_and_fuel(deck):
    for alt, M in [(0.0, 0.0), (10000.0, 0.8)]:
        assert deck.thrust(alt, M, 0.0) < deck.thrust(alt, M, 1.0)
        assert deck.fuel_flow(alt, M, 0.0) < deck.fuel_flow(alt, M, 1.0)


def test_inputs_are_clipped(deck):
    assert deck.thrust(20000.0, 2.0, 1.5) == deck.thrust(10000.0, 0.8, 1.0)



def test_blended_deck_endpoints_and_midpoint(deck):
    from jet_uav_sim.thrust_deck import BlendedDeck
    other = ThrustDeck(deck.df.assign(thrust_N=deck.df.thrust_N * 0.9))
    args = (10000.0, 0.8, 1.0)
    assert BlendedDeck(deck, other, 0.0).thrust(*args) == deck.thrust(*args)
    assert BlendedDeck(deck, other, 1.0).thrust(*args) == pytest.approx(other.thrust(*args))
    assert BlendedDeck(deck, other, 0.5).thrust(*args) == pytest.approx(0.95 * deck.thrust(*args))