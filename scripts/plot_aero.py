"""
Aerodynamic model plots.

    python scripts/plot_aero.py

Outputs:
    results/aero_drag_polar.png
    results/aero_drag_rise.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.aero import AeroModel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

aero = AeroModel()
print(f"CD0 = {aero.cd0:.4f}, e = {aero.oswald:.3f}, K = {aero.k_induced:.4f}")

# 1) Drag polars ----------------------------------------------------------------
cl = np.linspace(0.0, 1.0, 101)
fig, ax = plt.subplots(figsize=(8, 5))
for M in (0.3, 0.6, 0.8, 0.85, 0.9):
    ax.plot(aero.cd(cl, M), cl, label=f"M {M:.2f}")
cl_star = np.sqrt(aero.cd0 / aero.k_induced)
ax.plot(aero.cd(cl_star, 0.3), cl_star, "k*", ms=12, label="(L/D)max")
ax.set_xlabel("CD")
ax.set_ylabel("CL")
ax.set_title("Drag polars")
ax.set_xlim(0, 0.15)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "aero_drag_polar.png", dpi=150)
plt.close(fig)

# 2) Drag rise with Mach ----------------------------------------------------------
mach = np.linspace(0.3, 0.98, 200)
fig, ax = plt.subplots(figsize=(8, 5))
for c in (0.1, 0.2, 0.4):
    ax.plot(mach, aero.cd(c, mach), label=f"CL {c:.1f}")
    ax.axvline(aero.mach_dd(c), ls=":", lw=1, color=ax.lines[-1].get_color())
ax.set_xlabel("Mach number")
ax.set_ylabel("CD")
ax.set_title("Transonic drag rise (dotted: drag-divergence Mach, Korn)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "aero_drag_rise.png", dpi=150)
plt.close(fig)

print(f"Plots saved in {RESULTS}")