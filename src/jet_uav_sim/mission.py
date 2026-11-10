"""
Quasi-steady mission analysis.

Cruise / loiter : level-flight trim every dt, fuel burned, mass updated
Climb           : specific excess power Ps = (T - D) V / W at max dry thrust,
                  integrated in altitude slices at constant Mach
Descent         : idle thrust, Ps < 0 gives the sink rate

Each segment appends rows to a log: time, distance, altitude, Mach, mass, fuel used.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH, trim_level


@dataclass
class Mission:
    mass0: float
    fuel0: float
    aero: AeroModel = field(default_factory=AeroModel)
    deck: ThrustDeck = field(default_factory=lambda: ThrustDeck.from_csv(DECK_PATH))
    rows: list = field(default_factory=list)

    # --- state ---------------------------------------------------------------------
    def __post_init__(self):
        self.t, self.x, self.h, self.mass = 0.0, 0.0, 0.0, self.mass0
        self.fuel_used = 0.0
        self._log("start", 0.0)

    def _log(self, segment, mach):
        self.rows.append({"segment": segment, "t_s": self.t, "x_km": self.x / 1e3,
                          "h_m": self.h, "mach": mach, "mass_kg": self.mass,
                          "fuel_used_kg": self.fuel_used})

    def _burn(self, fuel):
        self.fuel_used += fuel
        self.mass -= fuel
        if self.fuel_used > self.fuel0:
            raise RuntimeError(f"out of fuel at t = {self.t / 60:.1f} min, "
                               f"x = {self.x / 1e3:.0f} km")

    def _level_drag(self, h, mach):
        _, _, rho, a = isa(h)
        V = mach * float(a)
        q_bar = 0.5 * float(rho) * V**2
        S = self.aero.airframe.wing.area
        cl = self.mass * G0 / (q_bar * S)
        return q_bar * S * float(self.aero.cd(cl, mach)), V

    # --- segments ------------------------------------------------------------------
    def fixed_fuel(self, name, fuel, duration_s=0.0):
        """Warm-up, take-off, landing allowances."""
        self.t += duration_s
        self._burn(fuel)
        self._log(name, 0.0)

    def climb(self, h_target, mach, dh=250.0, name="climb"):
        while self.h < h_target - 1e-6:
            step = min(dh, h_target - self.h)
            h_mid = self.h + 0.5 * step
            D, V = self._level_drag(h_mid, mach)
            T = self.deck.thrust(h_mid, mach, 1.0)
            Ps = (T - D) * V / (self.mass * G0)
            if Ps < 0.5:
                raise RuntimeError(f"cannot climb above {self.h:.0f} m at Mach {mach} "
                                   f"(Ps = {Ps:.2f} m/s)")
            dt = step / Ps
            self.t += dt
            self.x += V * dt
            self.h += step
            self._burn(self.deck.fuel_flow(h_mid, mach, 1.0) * dt)
            self._log(name, mach)

    def descend(self, h_target, mach, dh=250.0, name="descent"):
        while self.h > h_target + 1e-6:
            step = min(dh, self.h - h_target)
            h_mid = self.h - 0.5 * step
            D, V = self._level_drag(h_mid, mach)
            T = self.deck.thrust(h_mid, mach, 0.0)
            sink = (D - T) * V / (self.mass * G0)
            if sink < 0.5:
                raise RuntimeError(f"cannot descend at idle at {self.h:.0f} m, Mach {mach}")
            dt = step / sink
            self.t += dt
            self.x += V * dt
            self.h -= step
            self._burn(self.deck.fuel_flow(h_mid, mach, 0.0) * dt)
            self._log(name, mach)

    def cruise(self, mach, distance_m=None, duration_s=None, fuel_stop=None, dt=60.0,
               name="cruise"):
        """Level flight at the current altitude.

        Stops at whichever comes first: distance flown, duration, or total fuel used
        reaching fuel_stop [kg].
        """
        x_end = self.x + distance_m if distance_m is not None else np.inf
        t_end = self.t + duration_s if duration_s is not None else np.inf
        f_end = fuel_stop if fuel_stop is not None else np.inf
        while (self.x < x_end - 1e-6 and self.t < t_end - 1e-6
               and self.fuel_used < f_end - 1e-6):
            trim = trim_level(self.h, mach, self.mass, aero=self.aero, deck=self.deck)
            if not trim.converged:
                raise RuntimeError(f"cannot hold level flight at {self.h:.0f} m, Mach {mach}")
            V = mach * float(isa(self.h)[3])
            step = min(dt, (x_end - self.x) / V, t_end - self.t,
                       (f_end - self.fuel_used) / trim.fuel_flow)
            self.t += step
            self.x += V * step
            self._burn(trim.fuel_flow * step)
            self._log(name, mach)

    def log(self):
        return pd.DataFrame(self.rows)