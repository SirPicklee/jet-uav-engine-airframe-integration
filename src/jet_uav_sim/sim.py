"""
Closed-loop simulation: fixed-step RK4 with a digital autopilot (sample and hold).

The autopilot runs at the integration rate (default 50 Hz); its commands are
held constant during each step, as in a real flight computer.
"""

from dataclasses import dataclass

import numpy as np

from jet_uav_sim.atmosphere import isa
from jet_uav_sim.control import AltitudeHold, Autothrottle, HeadingHold
from jet_uav_sim.dynamics6dof import STATE_NAMES_6DOF, SixDofModel, state6_from_trim
from jet_uav_sim.dynamics import STATE_NAMES, LongitudinalModel, state_from_trim
from jet_uav_sim.trim import trim_level


def rk4_step(f, s, dt):
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass
class SimResult:
    t: np.ndarray
    states: np.ndarray          # (n, 8), columns as STATE_NAMES
    de: np.ndarray
    throttle_cmd: np.ndarray
    theta_cmd: np.ndarray
    V: np.ndarray
    mach: np.ndarray

    def __getitem__(self, name):
        return self.states[:, STATE_NAMES.index(name)]


def simulate_autopilot(altitude, mach, fuel, V_cmd_fn, h_cmd_fn, t_end, dt=0.02,
                       model=None):
    """Start from level trim and fly with autothrottle + altitude hold.

    V_cmd_fn(t), h_cmd_fn(t): commanded airspeed [m/s] and altitude [m].
    """
    model = model or LongitudinalModel()
    trim = trim_level(altitude, mach, model.mass(fuel), aero=model.aero, deck=model.deck)
    s = state_from_trim(trim, fuel)

    at = Autothrottle(throttle_trim=trim.throttle)
    ah = AltitudeHold(theta_trim=np.radians(trim.alpha_deg),
                      de_trim=np.radians(trim.de_deg))

    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    states = np.zeros((n, len(s)))
    de_log, thr_log, th_log, V_log, M_log = (np.zeros(n) for _ in range(5))

    for i in range(n):
        u, w, q, theta, x, h, fuel_i, thr = s
        V = np.hypot(u, w)
        h_dot = u * np.sin(theta) - w * np.cos(theta)

        thr_cmd = at.command(V_cmd_fn(t[i]), V, dt)
        de, theta_cmd = ah.command(h_cmd_fn(t[i]), h, h_dot, theta, q, dt)

        states[i] = s
        de_log[i], thr_log[i], th_log[i] = de, thr_cmd, theta_cmd
        V_log[i], M_log[i] = V, V / float(isa(h)[3])

        if i < n - 1:
            s = rk4_step(lambda st: model.derivatives(st, de, thr_cmd), s, dt)

    return SimResult(t, states, de_log, thr_log, th_log, V_log, M_log)



@dataclass
class Sim6Result:
    t: np.ndarray
    states: np.ndarray          # (n, 14), columns as STATE_NAMES_6DOF
    da: np.ndarray
    de: np.ndarray
    dr: np.ndarray
    throttle_cmd: np.ndarray
    phi_cmd: np.ndarray
    V: np.ndarray
    beta: np.ndarray

    def __getitem__(self, name):
        return self.states[:, STATE_NAMES_6DOF.index(name)]


def simulate_autopilot_6dof(altitude, mach, fuel, V_cmd_fn, h_cmd_fn, psi_cmd_fn, t_end,
                            dt=0.02, model=None):
    """6-DoF closed loop: autothrottle, altitude hold and heading hold at 1/dt Hz.

    In a steady turn the aircraft pitches at q = g tan(phi) sin(phi) cos(theta) / V;
    that part is removed from the pitch-rate feedback so the altitude hold does
    not fight the turn.
    """
    model = model or SixDofModel()
    trim = trim_level(altitude, mach, model.mass(fuel), aero=model.aero, deck=model.deck)
    s = state6_from_trim(trim, fuel)

    at = Autothrottle(throttle_trim=trim.throttle)
    ah = AltitudeHold(theta_trim=np.radians(trim.alpha_deg),
                      de_trim=np.radians(trim.de_deg))
    hh = HeadingHold()

    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    states = np.zeros((n, len(s)))
    logs = {k: np.zeros(n) for k in ("da", "de", "dr", "thr", "phi_cmd", "V", "beta")}

    for i in range(n):
        u, v, w, p, q, r, phi, theta, psi, x, y, h, fuel_i, thr = s
        V = np.sqrt(u**2 + v**2 + w**2)
        beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
        h_dot = u * np.sin(theta) - v * np.sin(phi) * np.cos(theta) - w * np.cos(phi) * np.cos(theta)
        q_turn = 9.80665 * np.tan(phi) * np.sin(phi) * np.cos(theta) / V

        thr_cmd = at.command(V_cmd_fn(t[i]), V, dt)
        de, _ = ah.command(h_cmd_fn(t[i]), h, h_dot, theta, q - q_turn, dt)
        da, dr, phi_cmd = hh.command(psi_cmd_fn(t[i]), psi, phi, theta, p, r, beta, V)

        states[i] = s
        for k, val in zip(logs, (da, de, dr, thr_cmd, phi_cmd, V, beta)):
            logs[k][i] = val

        if i < n - 1:
            s = rk4_step(lambda st: model.derivatives(st, da, de, dr, thr_cmd), s, dt)

    return Sim6Result(t, states, logs["da"], logs["de"], logs["dr"], logs["thr"],
                      logs["phi_cmd"], logs["V"], logs["beta"])