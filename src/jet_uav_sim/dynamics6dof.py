"""
Six-degree-of-freedom rigid-body dynamics with the engine and fuel burn.

Flat, non-rotating Earth; body axes; Euler angles (phi, theta, psi);
thrust along the body x-axis through the CG; products of inertia neglected.

State  : u, v, w [m/s]  p, q, r [rad/s]  phi, theta, psi [rad]
         x (north), y (east), h [m]  fuel [kg]  throttle [-]
Inputs : aileron da, elevator de, rudder dr [rad], throttle command [-]

Force equations (body axes)
    du/dt = r v - q w + X/m - g sin(theta)
    dv/dt = p w - r u + Y/m + g cos(theta) sin(phi)
    dw/dt = q u - p v + Z/m + g cos(theta) cos(phi)
Moment equations (Ixz = 0)
    dp/dt = (L + (Iyy - Izz) q r) / Ixx
    dq/dt = (M + (Izz - Ixx) p r) / Iyy
    dr/dt = (N + (Ixx - Iyy) p q) / Izz
Kinematics
    dphi/dt   = p + (q sin(phi) + r cos(phi)) tan(theta)
    dtheta/dt = q cos(phi) - r sin(phi)
    dpsi/dt   = (q sin(phi) + r cos(phi)) / cos(theta)

Longitudinal forces are the same as in the 3-DoF model (small-sideslip
approximation), so with the lateral states at zero both models agree exactly.
"""

from dataclasses import dataclass, field

import numpy as np

from jet_uav_sim.aero import AeroModel, LateralAero
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH, TrimResult

STATE_NAMES_6DOF = ["u", "v", "w", "p", "q", "r", "phi", "theta", "psi",
                    "x", "y", "h", "fuel", "throttle"]


@dataclass
class SixDofModel:
    aero: AeroModel = field(default_factory=AeroModel)
    lateral: LateralAero = field(default_factory=LateralAero)
    deck: ThrustDeck = field(default_factory=lambda: ThrustDeck.from_csv(DECK_PATH))
    tau_engine: float = 1.02     # s, fitted to turbofan-cycle (scripts/fit_engine_lag.py)

    def mass(self, fuel):
        m = self.aero.airframe.mass
        return m.empty + m.payload + max(fuel, 0.0)

    def derivatives(self, state, da, de, dr, throttle_cmd):
        u, v, w, p, q, r, phi, theta, psi, x, y, h, fuel, thr = state
        af = self.aero.airframe
        S, c, b = af.wing.area, af.wing.mac, af.wing.span

        _, _, rho, a = isa(h)
        rho, a = float(rho), float(a)
        V = np.sqrt(u**2 + v**2 + w**2)
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
        mach = V / a
        q_bar = 0.5 * rho * V**2
        q_hat = q * c / (2 * V)
        p_hat, r_hat = p * b / (2 * V), r * b / (2 * V)

        # Aerodynamics
        cl = self.aero.cl(alpha, mach, de)
        L_aero = q_bar * S * cl
        D = q_bar * S * float(self.aero.cd(cl, mach))
        Y = q_bar * S * self.lateral.c_side(beta, dr)
        L_roll = q_bar * S * b * self.lateral.c_roll(beta, p_hat, r_hat, da, dr)
        M_pitch = q_bar * S * c * self.aero.cm(alpha, mach, de, q_hat)
        N_yaw = q_bar * S * b * self.lateral.c_yaw(beta, p_hat, r_hat, da, dr)

        # Engine
        thr_c = np.clip(thr, 0.0, 1.0)
        T = self.deck.thrust(h, mach, thr_c) if fuel > 0 else 0.0
        wf = self.deck.fuel_flow(h, mach, thr_c) if fuel > 0 else 0.0

        # Wind axes -> body axes (small sideslip)
        X = T - D * np.cos(alpha) + L_aero * np.sin(alpha)
        Z = -D * np.sin(alpha) - L_aero * np.cos(alpha)

        m = self.mass(fuel)
        Ixx, Iyy, Izz = af.inertia(m)

        sphi, cphi = np.sin(phi), np.cos(phi)
        sth, cth = np.sin(theta), np.cos(theta)
        spsi, cpsi = np.sin(psi), np.cos(psi)

        du = r * v - q * w + X / m - G0 * sth
        dv = p * w - r * u + Y / m + G0 * cth * sphi
        dw = q * u - p * v + Z / m + G0 * cth * cphi
        dp = (L_roll + (Iyy - Izz) * q * r) / Ixx
        dq = (M_pitch + (Izz - Ixx) * p * r) / Iyy
        dr_ = (N_yaw + (Ixx - Iyy) * p * q) / Izz
        dphi = p + (q * sphi + r * cphi) * np.tan(theta)
        dtheta = q * cphi - r * sphi
        dpsi = (q * sphi + r * cphi) / cth

        # Position: body velocity rotated to north-east-down, h = -down
        dx = (u * cth * cpsi + v * (sphi * sth * cpsi - cphi * spsi)
              + w * (cphi * sth * cpsi + sphi * spsi))
        dy = (u * cth * spsi + v * (sphi * sth * spsi + cphi * cpsi)
              + w * (cphi * sth * spsi - sphi * cpsi))
        dh = u * sth - v * sphi * cth - w * cphi * cth

        return np.array([
            du, dv, dw, dp, dq, dr_, dphi, dtheta, dpsi, dx, dy, dh,
            -wf,
            (np.clip(throttle_cmd, 0.0, 1.0) - thr) / self.tau_engine,
        ])


def state6_from_trim(t: TrimResult, fuel, heading=0.0):
    """Initial 6-DoF state for wings-level steady flight at a trim point."""
    V = t.mach * float(isa(t.altitude)[3])
    alpha = np.radians(t.alpha_deg)
    return np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha), 0.0, 0.0, 0.0,
                     0.0, alpha, heading, 0.0, 0.0, t.altitude, fuel, t.throttle])