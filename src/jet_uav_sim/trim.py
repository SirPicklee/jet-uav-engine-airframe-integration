"""
Steady level flight trim: engine and airframe solved together.

Unknowns: angle of attack alpha, elevator de, throttle.
Equations (thrust along the body x-axis, through the CG):
    L + T sin(alpha) = W      vertical force balance
    T cos(alpha)     = D      horizontal force balance
    Cm               = 0      pitching moment balance
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.thrust_deck import ThrustDeck

ROOT = Path(__file__).resolve().parents[2]
DECK_PATH = ROOT / "data" / "thrust_deck.csv"


@dataclass
class TrimResult:
    altitude: float
    mach: float
    mass: float
    alpha_deg: float
    de_deg: float
    throttle: float
    thrust: float        # N
    drag: float          # N
    fuel_flow: float     # kg/s
    cl: float
    lift_to_drag: float
    converged: bool


def trim_level(altitude, mach, mass, aero=None, deck=None):
    """Trim for steady level flight. converged=False if the engine cannot hold it."""
    aero = aero or AeroModel()
    deck = deck or ThrustDeck.from_csv(DECK_PATH)
    _, _, rho, a = isa(altitude)
    V = mach * float(a)
    q_bar = 0.5 * float(rho) * V**2
    S = aero.airframe.wing.area
    W = mass * G0

    def residuals(x):
        alpha, de, thr = x
        cl = aero.cl(alpha, mach, de)
        L = q_bar * S * cl
        D = q_bar * S * aero.cd(cl, mach)
        T = deck.thrust(altitude, mach, thr)
        return [
            (L + T * np.sin(alpha) - W) / W,
            (T * np.cos(alpha) - D) / W,
            aero.cm(alpha, mach, de),
        ]

    sol = least_squares(
        residuals,
        x0=[np.radians(3.0), 0.0, 0.5],
        bounds=([np.radians(-5), np.radians(-25), 0.0],
                [np.radians(20), np.radians(25), 1.0]),
        xtol=1e-12, ftol=1e-12,
    )
    alpha, de, thr = sol.x
    cl = aero.cl(alpha, mach, de)
    D = q_bar * S * aero.cd(cl, mach)
    return TrimResult(
        altitude=altitude, mach=mach, mass=mass,
        alpha_deg=np.degrees(alpha), de_deg=np.degrees(de), throttle=thr,
        thrust=deck.thrust(altitude, mach, thr), drag=D,
        fuel_flow=deck.fuel_flow(altitude, mach, thr),
        cl=cl, lift_to_drag=cl / aero.cd(cl, mach),
        converged=bool(np.max(np.abs(sol.fun)) < 1e-6),
    )