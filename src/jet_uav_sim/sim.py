"""
Closed-loop simulation: fixed-step RK4 with a digital autopilot (sample and hold).

The autopilot runs at the integration rate (default 50 Hz); its commands are
held constant during each step, as in a real flight computer.
"""

from dataclasses import dataclass

import numpy as np

from jet_uav_sim.atmosphere import isa
from jet_uav_sim.control import AltitudeHold, Autothrottle
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