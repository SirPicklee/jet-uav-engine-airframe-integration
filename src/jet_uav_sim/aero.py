"""
Longitudinal aerodynamics of the jet UAV from conceptual design methods.

  Lift slope      : DATCOM / Helmbold formula with Prandtl-Glauert correction
  Zero-lift drag  : equivalent skin friction  CD0 = Cfe * Swet / Sref
  Induced drag    : Oswald factor for swept wings (Raymer)
  Wave drag       : Korn equation for drag divergence + Lock's 4th-power law
  Pitching moment : static margin and pitch damping estimate

References: Raymer, Aircraft Design: A Conceptual Approach (6th ed.);
            USAF DATCOM; Mason, Transonic Aerodynamics of Airfoils and Wings.
"""

from dataclasses import dataclass, field

import numpy as np

from jet_uav_sim.airframe import UAV, Airframe


@dataclass
class AeroModel:
    airframe: Airframe = field(default_factory=lambda: UAV)
    cfe: float = 0.0035           # equivalent skin friction coefficient (Raymer, fighter)
    swet_sref: float = 4.0        # wetted area / reference area
    korn_kappa: float = 0.87      # airfoil technology factor (0.87 conventional, 0.95 supercritical)
    alpha0_deg: float = -1.0      # zero-lift angle of attack
    cl_max: float = 1.2           # clean, low speed
    static_margin: float = 0.08   # (x_np - x_cg) / MAC
    cm0: float = 0.02
    cm_de: float = -0.9           # per rad of elevator
    cl_de: float = 0.4            # per rad of elevator
    cmq_factor: float = -1.1      # Cmq = factor * CLa * (tail arm / MAC)^2, rough
    cmq_factor: float = -1.1      # Cmq = factor * CLa * (tail arm / MAC)^2, rough
    eta: float = 0.95             # airfoil efficiency (DATCOM)
    oswald_factor: float = 1.0    # scales the Raymer estimate (uncertainty studies)

    # --- lift ------------------------------------------------------------------
    def cl_alpha(self, mach):
        """Wing lift-curve slope [1/rad] (DATCOM / Helmbold)."""
        w = self.airframe.wing
        A = w.aspect_ratio
        beta2 = max(1.0 - mach**2, 0.05)
        tan_half = np.tan(np.radians(w.sweep_deg(0.5)))
        return 2 * np.pi * A / (2 + np.sqrt(4 + A**2 * beta2 / self.eta**2
                                            * (1 + tan_half**2 / beta2)))

    def cl(self, alpha, mach, de=0.0):
        a0 = np.radians(self.alpha0_deg)
        return self.cl_alpha(mach) * (alpha - a0) + self.cl_de * de

    # --- drag ------------------------------------------------------------------
    @property
    def cd0(self):
        return self.cfe * self.swet_sref

    @property
    def oswald(self):
        """Oswald efficiency for swept wings (Raymer eq. 12.49)."""
        w = self.airframe.wing
        A, sweep = w.aspect_ratio, np.radians(w.sweep_le_deg)
        return 4.61 * (1 - 0.045 * A**0.68) * np.cos(sweep) ** 0.15 - 3.1

    @property
    def k_induced(self):
        return 1.0 / (np.pi * self.airframe.wing.aspect_ratio * self.oswald * self.oswald_factor)

    def mach_dd(self, cl):
        """Drag-divergence Mach number (Korn equation, swept form)."""
        w = self.airframe.wing
        c = np.cos(np.radians(w.sweep_deg(0.25)))
        return self.korn_kappa / c - w.thickness / c**2 - cl / (10 * c**3)

    def cd_wave(self, cl, mach):
        """Wave drag: Lock's 4th-power law above the critical Mach number."""
        m_crit = self.mach_dd(cl) - (0.1 / 80) ** (1 / 3)
        return np.where(mach > m_crit, 20 * (mach - m_crit) ** 4, 0.0)

    def cd(self, cl, mach):
        return self.cd0 + self.k_induced * cl**2 + self.cd_wave(cl, mach)

    # --- pitching moment ---------------------------------------------------------
    def cm_alpha(self, mach):
        return -self.static_margin * self.cl_alpha(mach)

    def cm_q(self, mach):
        """Pitch damping per rad of q * MAC / (2V)."""
        arm = 0.35 * self.airframe.length / self.airframe.wing.mac
        return self.cmq_factor * self.cl_alpha(mach) * arm**2

    def cm(self, alpha, mach, de=0.0, q_hat=0.0):
        a0 = np.radians(self.alpha0_deg)
        return (self.cm0 + self.cm_alpha(mach) * (alpha - a0)
                + self.cm_de * de + self.cm_q(mach) * q_hat)