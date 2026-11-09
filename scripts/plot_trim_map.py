"""
Trim map over the flight envelope: where the UAV can fly level, and how efficiently.

    python scripts/plot_trim_map.py

Outputs:
    results/trim_map_specific_range.png
    results/trim_map_throttle.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.atmosphere import isa
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH, trim_level

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MASS = 6000.0
altitudes = np.arange(0.0, 12001.0, 1000.0)
machs = np.round(np.arange(0.20, 0.901, 0.05), 2)

aero = AeroModel()
deck = ThrustDeck.from_csv(DECK_PATH)

shape = (len(altitudes), len(machs))
spec_range = np.full(shape, np.nan)   # km per kg of fuel
throttle = np.full(shape, np.nan)

for i, h in enumerate(altitudes):
    a = float(isa(h)[3])
    for j, M in enumerate(machs):
        t = trim_level(h, M, MASS, aero=aero, deck=deck)
        if not t.converged or t.cl > aero.cl_max:
            continue                      # thrust or lift limited: outside envelope
        speed_kmh = M * a * 3.6
        spec_range[i, j] = speed_kmh / (t.fuel_flow * 3600)
        throttle[i, j] = t.throttle
    print(f"  alt {h:6.0f} m done")

k = np.nanargmax(spec_range)
i_best, j_best = np.unravel_index(k, shape)
print(f"Best specific range: {spec_range[i_best, j_best]:.2f} km/kg "
      f"at {altitudes[i_best]:.0f} m, Mach {machs[j_best]:.2f}")

M_grid, H_grid = np.meshgrid(machs, altitudes / 1000)


def envelope_plot(z, label, title, fname, levels):
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(M_grid, H_grid, z, levels=levels, cmap="viridis")
    ax.contour(M_grid, H_grid, z, levels=levels, colors="k", linewidths=0.3)
    fig.colorbar(cs, label=label)
    ax.plot(machs[j_best], altitudes[i_best] / 1000, "r*", ms=15,
            label="best specific range")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(f"{title} (mass {MASS:.0f} kg, white: cannot hold level flight)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


envelope_plot(spec_range, "Specific range [km/kg]", "Specific range",
              "trim_map_specific_range.png", 15)
envelope_plot(throttle, "Throttle [-]", "Level flight throttle",
              "trim_map_throttle.png", np.linspace(0, 1, 11))
print(f"Plots saved in {RESULTS}")