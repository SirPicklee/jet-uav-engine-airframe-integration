"""
Numerical linearization about a trim point and modal analysis.

Reduced state (x and fuel removed: they do not feed back into the dynamics
on short time scales):
    [u, w, q, theta, h, throttle]
Inputs:
    [de, throttle_cmd]

A and B are built with central differences: dx/dt ~ A dx + B du.
"""

import numpy as np

STATE_IDX = [0, 1, 2, 3, 5, 7]           # u, w, q, theta, h, throttle in the full state
REDUCED_NAMES = ["u", "w", "q", "theta", "h", "throttle"]


def linearize(model, state0, de0, thr_cmd0, eps=1e-5):
    """Jacobians A (6x6) and B (6x2) of the longitudinal model."""
    def f(xr, de, thr_cmd):
        s = state0.copy()
        s[STATE_IDX] = xr
        return model.derivatives(s, de, thr_cmd)[STATE_IDX]

    x0 = state0[STATE_IDX]
    n = len(x0)
    A = np.zeros((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(x0[j]))
        dx = np.zeros(n)
        dx[j] = h
        A[:, j] = (f(x0 + dx, de0, thr_cmd0) - f(x0 - dx, de0, thr_cmd0)) / (2 * h)

    B = np.zeros((n, 2))
    h = 1e-5
    B[:, 0] = (f(x0, de0 + h, thr_cmd0) - f(x0, de0 - h, thr_cmd0)) / (2 * h)
    B[:, 1] = (f(x0, de0, thr_cmd0 + h) - f(x0, de0, thr_cmd0 - h)) / (2 * h)
    return A, B


def modes(A):
    """Oscillatory modes (one per complex pair) and real modes, from A's eigenvalues.

    Returns two lists of dicts sorted by natural frequency:
      oscillatory: eigenvalue, wn [rad/s], zeta, period [s]
      real       : eigenvalue, time constant [s] (negative means unstable)
    """
    eig = np.linalg.eigvals(A)
    osc, real = [], []
    for lam in eig:
        if abs(lam.imag) > 1e-9:
            if lam.imag > 0:
                wn = abs(lam)
                osc.append({"eigenvalue": lam, "wn": wn, "zeta": -lam.real / wn,
                            "period": 2 * np.pi / lam.imag})
        else:
            lr = lam.real
            real.append({"eigenvalue": lr,
                         "time_constant": -1.0 / lr if abs(lr) > 1e-12 else np.inf})
    osc.sort(key=lambda m: m["wn"])
    real.sort(key=lambda m: abs(m["eigenvalue"]))
    return osc, real


LAT_IDX = [1, 3, 5, 6]                    # v, p, r, phi in the 6-DoF state
LAT_NAMES = ["v", "p", "r", "phi"]


def linearize_lateral(model, state0, controls0, eps=1e-5):
    """Lateral-directional A (4x4) and B (4x2: aileron, rudder) of the 6-DoF model.

    The longitudinal states are held at the trim point (the usual decoupling).
    controls0 = (da, de, dr, throttle_cmd) at trim.
    """
    da0, de0, dr0, thr0 = controls0

    def f(xr, da, dr):
        s = state0.copy()
        s[LAT_IDX] = xr
        return model.derivatives(s, da, de0, dr, thr0)[LAT_IDX]

    x0 = state0[LAT_IDX]
    n = len(x0)
    A = np.zeros((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(x0[j]))
        dx = np.zeros(n)
        dx[j] = h
        A[:, j] = (f(x0 + dx, da0, dr0) - f(x0 - dx, da0, dr0)) / (2 * h)

    h = 1e-5
    B = np.zeros((n, 2))
    B[:, 0] = (f(x0, da0 + h, dr0) - f(x0, da0 - h, dr0)) / (2 * h)
    B[:, 1] = (f(x0, da0, dr0 + h) - f(x0, da0, dr0 - h)) / (2 * h)
    return A, B