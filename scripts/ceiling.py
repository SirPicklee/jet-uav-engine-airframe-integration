"""
Climb performance and ceilings at max dry thrust.

  Rate of climb (steady, small flight-path angle): RC = (T - D) V / W
  Service ceiling  : max RC = 0.5 m/s (100 ft/min)
  Absolute ceiling : max RC = 0

    python scripts/ceiling.py

Output: results/ceiling.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.airframe import UAV
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

aero = AeroModel()
deck = ThrustDeck.from_csv(DECK_PATH)
S = UAV.wing.area

altitudes = np.arange(0.0, 20001.0, 250.0)
machs = np.linspace(0.2, 0.9, 141)
masses = {"MTOW": UAV.mass.mtow, "half fuel": UAV.mass.mtow - 0.5 * UAV.mass.fuel_max}


H_TOP = 15000.0                      # top of the thrust deck
P_TOP = float(isa(H_TOP)[1])


def thrust_max(h, M):
    """Max dry thrust; above the deck it is scaled with ambient pressure.

    In the isothermal layer (11-20 km) the inlet temperature at a given Mach is
    constant, so the engine runs at the same corrected point: F/delta is constant
    and thrust scales exactly with pressure.
    """
    if h <= H_TOP:
        return deck.thrust(h, M, 1.0)
    return deck.thrust(H_TOP, M, 1.0) * float(isa(h)[1]) / P_TOP

def max_climb(h, mass):
    """Best rate of climb [m/s] over Mach at altitude h, and the Mach for it."""
    _, _, rho, a = isa(h)
    rho, a = float(rho), float(a)
    best_rc, best_m = -np.inf, np.nan
    for M in machs:
        V = M * a
        q_bar = 0.5 * rho * V**2
        cl = mass * G0 / (q_bar * S)
        if cl > aero.cl_max:
            continue
        D = q_bar * S * float(aero.cd(cl, M))
        rc = (thrust_max(h, M) - D) * V / (mass * G0)
        if rc > best_rc:
            best_rc, best_m = rc, M
    return best_rc, best_m


def ceiling(h, rc, level):
    """Altitude where rc falls to level (linear interpolation), None if not reached."""
    below = np.where(rc < level)[0]
    if len(below) == 0:
        return None
    k = below[0]
    return h[k - 1] + (level - rc[k - 1]) * (h[k] - h[k - 1]) / (rc[k] - rc[k - 1])


fig, ax = plt.subplots(figsize=(8, 6))
for label, mass in masses.items():
    rc, mach_best = np.array([max_climb(h, mass) for h in altitudes]).T
    crz = ceiling(altitudes, rc, 1.5)
    svc, absl = ceiling(altitudes, rc, 0.5), ceiling(altitudes, rc, 0.0)
    print(f"{label} ({mass:.0f} kg):")
    print(f"  sea-level max rate of climb {rc[0]:.1f} m/s at Mach {mach_best[0]:.2f}")
    print(f"  cruise ceiling   {crz:.0f} m" if crz else "  cruise ceiling   above 20 km")
    print(f"  service ceiling  {svc:.0f} m" if svc else "  service ceiling  above 20 km")
    print(f"  absolute ceiling {absl:.0f} m" if absl else "  absolute ceiling above 20 km")
    ax.plot(rc, altitudes / 1000, lw=2, label=label)
    if svc:
        ax.plot(0.5, svc / 1000, "o", color=ax.lines[-1].get_color())
ax.axvline(0.5, color="k", ls=":", lw=1, label="0.5 m/s (service ceiling)")
ax.set_xlabel("Max rate of climb [m/s]")
ax.set_ylabel("Altitude [km]")
ax.set_title("Climb performance at max dry thrust")
ax.set_xlim(left=0)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "ceiling.png", dpi=150)
print(f"Saved {RESULTS / 'ceiling.png'}")