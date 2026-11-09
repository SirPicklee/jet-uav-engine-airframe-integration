"""
Notional jet UAV airframe: geometry, mass and inertia.

Single-engine, ~6 t class, trapezoidal (cropped delta) wing.
Values are a conceptual design, not data of any real aircraft.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Wing:
    area: float = 30.0          # m^2, reference area S
    span: float = 10.0          # m
    taper: float = 0.25         # tip chord / root chord
    sweep_le_deg: float = 40.0  # leading-edge sweep
    thickness: float = 0.05     # thickness-to-chord ratio t/c

    @property
    def aspect_ratio(self):
        return self.span**2 / self.area

    @property
    def root_chord(self):
        return 2.0 * self.area / (self.span * (1.0 + self.taper))

    @property
    def mac(self):
        """Mean aerodynamic chord [m]."""
        lam, cr = self.taper, self.root_chord
        return 2.0 / 3.0 * cr * (1 + lam + lam**2) / (1 + lam)

    def sweep_deg(self, x_over_c):
        """Sweep of the line at fraction x_over_c of the chord [deg]."""
        tan_le = np.tan(np.radians(self.sweep_le_deg))
        tan_x = tan_le - 4.0 / self.aspect_ratio * x_over_c * (1 - self.taper) / (1 + self.taper)
        return np.degrees(np.arctan(tan_x))


@dataclass
class Mass:
    empty: float = 3500.0      # kg, operating empty mass
    payload: float = 700.0     # kg
    fuel_max: float = 1800.0   # kg, internal fuel

    @property
    def mtow(self):
        return self.empty + self.payload + self.fuel_max


@dataclass
class Airframe:
    wing: Wing
    mass: Mass
    length: float = 12.0       # m, fuselage length

    def inertia(self, total_mass):
        """Ixx, Iyy, Izz [kg m^2] from nondimensional radii of gyration.

        I = m * (R * L / 2)^2  (Roskam, Airplane Design Part V; fighter-type values).
        """
        b, L = self.wing.span, self.length
        Rx, Ry, Rz = 0.25, 0.38, 0.52
        Ixx = total_mass * (Rx * b / 2) ** 2
        Iyy = total_mass * (Ry * L / 2) ** 2
        Izz = total_mass * (Rz * (b + L) / 4) ** 2
        return Ixx, Iyy, Izz


UAV = Airframe(wing=Wing(), mass=Mass())