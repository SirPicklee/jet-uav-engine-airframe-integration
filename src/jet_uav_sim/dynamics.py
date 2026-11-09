"""
Longitudinal (3-DoF) rigid-body dynamics with the engine and fuel burn.

Flat, non-rotating Earth; body axes; thrust along the body x-axis through the CG.

State  : u, w [m/s], q [rad/s], theta [rad], x [m], h [m], fuel [kg], throttle [-]
Inputs : elevator de [rad], throttle command [-]

    du/dt = X/m - g sin(theta) - q w
    dw/dt = Z/m + g cos(theta) + q u
    dq/dt = M / Iyy
    dtheta/dt = q
    dx/dt = u cos(theta) + w sin(theta)
    dh/dt = u sin(theta) - w cos(theta)
    dfuel/dt = -fuel_flow
    dthrottle/dt = (command - throttle) / tau_engine   (spool lag)
"""

from dataclasses import dataclass, field

import numpy as np

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH, TrimResult

STATE_NAMES = ["u", "w", "q", "theta", "x", "h", "fuel", "throttle"]


@dataclass
class LongitudinalModel:
    aero: AeroModel = field(default_factory=AeroModel)
    deck: ThrustDeck = field(default_factory=lambda: ThrustDeck.from_csv(DECK_PATH))
    tau_engine: float = 1.0      # s, first-order spool lag (to be fitted to turbofan-cycle)

    def mass(self, fuel):
        m = self.aero.airframe.mass
        return m.empty + m.payload + max(fuel, 0.0)

    def derivatives(self, state, de, throttle_cmd):
        u, w, q, theta, x, h, fuel, thr = state
        af = self.aero.airframe
        S, c = af.wing.area, af.wing.mac

        _, _, rho, a = isa(h)
        rho, a = float(rho), float(a)
        V = np.hypot(u, w)
        alpha = np.arctan2(w, u)
        mach = V / a
        q_bar = 0.5 * rho * V**2
        q_hat = q * c / (2 * V)

        cl = self.aero.cl(alpha, mach, de)
        L = q_bar * S * cl
        D = q_bar * S * float(self.aero.cd(cl, mach))
        My = q_bar * S * c * self.aero.cm(alpha, mach, de, q_hat)

        thr_c = np.clip(thr, 0.0, 1.0)
        T = self.deck.thrust(h, mach, thr_c) if fuel > 0 else 0.0
        wf = self.deck.fuel_flow(h, mach, thr_c) if fuel > 0 else 0.0

        # wind axes -> body axes
        X = T - D * np.cos(alpha) + L * np.sin(alpha)
        Z = -D * np.sin(alpha) - L * np.cos(alpha)

        m = self.mass(fuel)
        _, Iyy, _ = af.inertia(m)

        return np.array([
            X / m - G0 * np.sin(theta) - q * w,
            Z / m + G0 * np.cos(theta) + q * u,
            My / Iyy,
            q,
            u * np.cos(theta) + w * np.sin(theta),
            u * np.sin(theta) - w * np.cos(theta),
            -wf,
            (np.clip(throttle_cmd, 0.0, 1.0) - thr) / self.tau_engine,
        ])


def state_from_trim(t: TrimResult, fuel):
    """Initial state for steady level flight at a trim point."""
    a = float(isa(t.altitude)[3])
    V = t.mach * a
    alpha = np.radians(t.alpha_deg)
    return np.array([V * np.cos(alpha), V * np.sin(alpha), 0.0, alpha,
                     0.0, t.altitude, fuel, t.throttle])