"""
Best loiter condition: minimum fuel flow over altitude and Mach.

    python scripts/find_loiter_optimum.py

Uses a mid-mission mass (half fuel). Output: results/loiter_fuel_flow.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.airframe import UAV
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH, trim_level

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MASS = UAV.mass.mtow - 0.5 * UAV.mass.fuel_max
altitudes = np.arange(0.0, 12001.0, 1000.0)
machs = np.round(np.arange(0.25, 0.801, 0.025), 3)

aero = AeroModel()
deck = ThrustDeck.from_csv(DECK_PATH)

wf = np.full((len(altitudes), len(machs)), np.nan)    # kg/h
ld = np.full_like(wf, np.nan)
for i, h in enumerate(altitudes):
    for j, M in enumerate(machs):
        t = trim_level(h, M, MASS, aero=aero, deck=deck)
        if t.converged and t.cl <= aero.cl_max:
            wf[i, j] = t.fuel_flow * 3600
            ld[i, j] = t.lift_to_drag

i, j = np.unravel_index(np.nanargmin(wf), wf.shape)
ld_max = 0.5 / np.sqrt(aero.cd0 * aero.k_induced)
print(f"Mass {MASS:.0f} kg")
print(f"Minimum fuel flow: {wf[i, j]:.0f} kg/h at {altitudes[i]:.0f} m, Mach {machs[j]:.3f}")
print(f"L/D there: {ld[i, j]:.2f}  (theoretical (L/D)max = {ld_max:.2f})")
print(f"Endurance on 1 t of fuel: {1000 / wf[i, j]:.1f} h")

print("\nBest loiter Mach at each altitude:")
for k, h in enumerate(altitudes):
    if np.all(np.isnan(wf[k])):
        continue
    jj = np.nanargmin(wf[k])
    print(f"  {h:6.0f} m   Mach {machs[jj]:.3f}   {wf[k, jj]:5.0f} kg/h   L/D {ld[k, jj]:5.2f}")

M_grid, H_grid = np.meshgrid(machs, altitudes / 1000)
fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(M_grid, H_grid, wf, levels=20, cmap="viridis_r")
ax.contour(M_grid, H_grid, wf, levels=20, colors="k", linewidths=0.3)
fig.colorbar(cs, label="Fuel flow [kg/h]")
ax.plot(machs[j], altitudes[i] / 1000, "r*", ms=15, label="minimum fuel flow")
ax.set_xlabel("Mach number")
ax.set_ylabel("Altitude [km]")
ax.set_title(f"Level-flight fuel flow, mass {MASS:.0f} kg")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "loiter_fuel_flow.png", dpi=150)
print(f"\nSaved {RESULTS / 'loiter_fuel_flow.png'}")