"""
Autopilot test: airspeed step, then altitude step, at 6 km.

    python scripts/sim_autopilot.py

Output: results/sim_autopilot.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.atmosphere import isa
from jet_uav_sim.sim import simulate_autopilot

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALT, MACH, FUEL = 6000.0, 0.6, 1800.0
V0 = MACH * float(isa(ALT)[3])
DV, DH = 20.0, 500.0           # speed step [m/s] at t = 10 s, altitude step [m] at t = 60 s
T_END = 300.0


def V_cmd(t):
    return V0 + (DV if t >= 10.0 else 0.0)


def h_cmd(t):
    return ALT + (DH if t >= 60.0 else 0.0)


r = simulate_autopilot(ALT, MACH, FUEL, V_cmd, h_cmd, T_END)

# Step metrics
spd = (r.t >= 10.0) & (r.t < 60.0)
alt = r.t >= 60.0
V_over = 100 * (r.V[spd].max() - (V0 + DV)) / DV
h_over = 100 * (r["h"][alt].max() - (ALT + DH)) / DH
h_dev_speed = np.abs(r["h"][spd] - ALT).max()
V_dev_alt = np.abs(r.V[alt] - (V0 + DV)).max()
print(f"Speed step  : overshoot {V_over:5.1f} %, max altitude deviation {h_dev_speed:5.1f} m")
print(f"Altitude step: overshoot {h_over:5.1f} %, max speed deviation {V_dev_alt:5.2f} m/s")
print(f"Final errors : V {r.V[-1] - (V0 + DV):+.2f} m/s, h {r['h'][-1] - (ALT + DH):+.1f} m")
print(f"Throttle range: {r.throttle_cmd.min():.2f} - {r.throttle_cmd.max():.2f}, "
      f"elevator range: {np.degrees(r.de.min()):.1f} to {np.degrees(r.de.max()):.1f} deg")

fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
axs[0].plot(r.t, r.V, label="V")
axs[0].plot(r.t, [V_cmd(t) for t in r.t], "k--", lw=1, label="command")
axs[0].set_ylabel("V [m/s]")
axs[1].plot(r.t, r["h"], label="h")
axs[1].plot(r.t, [h_cmd(t) for t in r.t], "k--", lw=1, label="command")
axs[1].set_ylabel("h [m]")
axs[2].plot(r.t, r.throttle_cmd, label="command")
axs[2].plot(r.t, r["throttle"], label="engine (lagged)")
axs[2].set_ylabel("throttle [-]")
axs[3].plot(r.t, np.degrees(r.de), label="elevator")
axs[3].plot(r.t, np.degrees(r["theta"]), label="pitch")
axs[3].plot(r.t, np.degrees(r.theta_cmd), "--", lw=1, label="pitch command")
axs[3].set_ylabel("angle [deg]")
axs[3].set_xlabel("time [s]")
for ax in axs:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axs[0].set_title(f"Autothrottle + altitude hold, {ALT / 1000:.0f} km, Mach {MACH}")
fig.tight_layout()
fig.savefig(RESULTS / "sim_autopilot.png", dpi=150)
print(f"Saved {RESULTS / 'sim_autopilot.png'}")