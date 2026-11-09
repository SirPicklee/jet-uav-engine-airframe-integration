"""
Elevator doublet from level trim: shows the short-period and phugoid modes.

    python scripts/sim_elevator_doublet.py

Output: results/sim_elevator_doublet.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from jet_uav_sim.atmosphere import G0
from jet_uav_sim.dynamics import LongitudinalModel, state_from_trim
from jet_uav_sim.trim import trim_level

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALT, MACH, FUEL = 6000.0, 0.6, 1800.0
T_END = 180.0
DOUBLET_DEG = 2.0

model = LongitudinalModel()
mass = model.mass(FUEL)
trim = trim_level(ALT, MACH, mass, aero=model.aero, deck=model.deck)
de_trim = np.radians(trim.de_deg)
print(f"Trim: alpha {trim.alpha_deg:.2f} deg, de {trim.de_deg:.2f} deg, "
      f"throttle {trim.throttle:.3f}")


def elevator(t):
    d = np.radians(DOUBLET_DEG)
    if 1.0 <= t < 2.0:
        return de_trim + d
    if 2.0 <= t < 3.0:
        return de_trim - d
    return de_trim


def rhs(t, s):
    return model.derivatives(s, elevator(t), trim.throttle)


sol = solve_ivp(rhs, (0.0, T_END), state_from_trim(trim, FUEL),
                max_step=0.02, rtol=1e-8, atol=1e-8)
t = sol.t
u, w, q, theta, x, h, fuel, thr = sol.y
V = np.hypot(u, w)
alpha = np.degrees(np.arctan2(w, u))

# Phugoid period from the speed peaks after the short period has died out
late = t > 10.0
dV = V[late] - V[late].mean()
crossings = t[late][1:][np.diff(np.sign(dV)) < 0]
if len(crossings) >= 2:
    T_ph = np.mean(np.diff(crossings))
    V0 = V[0]
    print(f"Phugoid period: {T_ph:.1f} s  (Lanchester estimate pi*sqrt(2)*V/g = "
          f"{np.pi * np.sqrt(2) * V0 / G0:.1f} s)")

fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
axs[0].plot(t, alpha)
axs[0].set_ylabel("alpha [deg]")
axs[1].plot(t, np.degrees(q))
axs[1].set_ylabel("q [deg/s]")
axs[2].plot(t, V)
axs[2].set_ylabel("V [m/s]")
axs[3].plot(t, h)
axs[3].set_ylabel("h [m]")
axs[3].set_xlabel("time [s]")
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].set_title(f"Elevator doublet ±{DOUBLET_DEG:.0f}°, "
                 f"{ALT / 1000:.0f} km, Mach {MACH}")

# zoomed inset on the short period
ins = axs[0].inset_axes([0.55, 0.35, 0.4, 0.55])
ins.plot(t, alpha)
ins.set_xlim(0, 8)
ins.set_title("short period", fontsize=8)
ins.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS / "sim_elevator_doublet.png", dpi=150)
print(f"Saved {RESULTS / 'sim_elevator_doublet.png'}")