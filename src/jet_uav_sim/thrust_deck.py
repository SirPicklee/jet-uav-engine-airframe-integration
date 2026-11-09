"""
Thrust deck: engine performance tabulated over altitude, Mach and throttle.

Throttle is defined between idle (0) and the max dry rating (1):
    T4 = T4_IDLE + throttle * (T4_rated - T4_IDLE)
where T4_rated is the T4 the rating allows at that flight condition.

The flight simulation reads the deck instead of calling the cycle solver,
which is far too slow to run inside a 6-DoF time step.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from jet_uav_sim.engine import T4_IDLE, make_engine, max_point, part_power_point

COLUMNS = ["altitude_m", "mach", "throttle", "T4_K", "thrust_N", "fuel_flow_kg_s",
           "N_L", "N_H", "egt_K", "limiter"]

def march_to(eng, altitude, mach, T4_from, T4_to, x, step=25.0):
    """Solve at T4_to starting from a converged point at T4_from.

    Tries one jump first. If the solver fails, walks from T4_from to T4_to
    in steps of at most `step` kelvin. Returns (MapPoint or None, x).
    """
    try:
        return part_power_point(eng, altitude, mach, T4_to, guess=x)
    except RuntimeError:
        pass
    n = int(np.ceil(abs(T4_to - T4_from) / step))
    x_try = x
    try:
        for T4 in np.linspace(T4_from, T4_to, n + 1)[1:]:
            p, x_try = part_power_point(eng, altitude, mach, T4, guess=x_try)
    except RuntimeError:
        return None, x
    return p, x_try

def generate(altitudes, machs, throttles, verbose=True):
    """Run the cycle over the grid. Returns a DataFrame with one row per point."""
    _, eng = make_engine()
    rows = []
    for alt in altitudes:
        x_max = None
        for mach in machs:
            p_max, limiter, x_max = max_point(eng, alt, mach, guess=x_max)
            T4_rated = p_max.result.design.T4
            x, T4_prev = x_max, T4_rated
            # march down from max power: each point starts from the previous one
            for thr in sorted(throttles, reverse=True):
                if thr >= 1.0:
                    p = p_max
                else:
                    T4 = T4_IDLE + thr * (T4_rated - T4_IDLE)
                    p, x = march_to(eng, alt, mach, T4_prev, T4, x)
                    if p is not None:
                        T4_prev = T4
                if p is None:
                    rows.append([alt, mach, thr] + [np.nan] * 6 + ["no_conv"])
                    continue
                r = p.result
                rows.append([alt, mach, thr, r.design.T4, r.net_thrust, r.fuel_flow,
                             p.N_L, p.N_H, p.egt, limiter if thr >= 1.0 else "throttle"])
            if verbose:
                print(f"  alt {alt:6.0f} m  M {mach:4.2f}  "
                      f"Fn_max {p_max.result.net_thrust / 1e3:5.1f} kN  [{limiter}]")
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["altitude_m", "mach", "throttle"]).reset_index(drop=True)


@dataclass
class ThrustDeck:
    """Interpolates thrust and fuel flow from a generated deck."""
    df: pd.DataFrame

    def __post_init__(self):
        df = self.df
        self.altitudes = np.sort(df.altitude_m.unique())
        self.machs = np.sort(df.mach.unique())
        self.throttles = np.sort(df.throttle.unique())
        shape = (len(self.altitudes), len(self.machs), len(self.throttles))
        if len(df) != np.prod(shape):
            raise ValueError("deck is not a full grid")
        if df[["thrust_N", "fuel_flow_kg_s"]].isna().any().any():
            raise ValueError("deck has unconverged points; fill them before use")
        grid = (self.altitudes, self.machs, self.throttles)
        self._thrust = RegularGridInterpolator(grid, df.thrust_N.to_numpy().reshape(shape))
        self._fuel = RegularGridInterpolator(grid, df.fuel_flow_kg_s.to_numpy().reshape(shape))

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path))

    def _clip(self, altitude, mach, throttle):
        return (np.clip(altitude, self.altitudes[0], self.altitudes[-1]),
                np.clip(mach, self.machs[0], self.machs[-1]),
                np.clip(throttle, self.throttles[0], self.throttles[-1]))

    def thrust(self, altitude, mach, throttle):
        """Net thrust [N]."""
        return float(self._thrust(self._clip(altitude, mach, throttle)))

    def fuel_flow(self, altitude, mach, throttle):
        """Fuel flow [kg/s]."""
        return float(self._fuel(self._clip(altitude, mach, throttle)))
