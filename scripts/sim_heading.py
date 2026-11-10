"""
6-DoF autopilot test: 90 deg heading change at 6 km, Mach 0.6,
with autothrottle and altitude hold engaged.

    python scripts/sim_heading.py

Output: results/sim_heading_turn.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.atmosphere import isa
from jet_uav_sim.sim import simulate_autopilot_6dof

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALT, MACH, FUEL = 6000.0, 0.6, 1800.0
V0 = MACH * float(isa(ALT)[3])
PSI_NEW = np.radians(90.0)
T_STEP, T_END = 5.0, 100.0

r = simulate_autopilot_6dof(ALT, MACH, FUEL,
                            V_cmd_fn=lambda t: V0,
                            h_cmd_fn=lambda t: ALT,
                            psi_cmd_fn=lambda t: PSI_NEW if t >= T_STEP else 0.0,
                            t_end=T_END)

psi = np.degrees(r["psi"])
phi = np.degrees(r["phi"])
err = np.abs(psi - 90.0)
settled = r.t[(r.t > T_STEP) & (err > 2.0)]
t_settle = settled[-1] - T_STEP if len(settled) else 0.0
turn_rate = np.degrees(9.80665 * np.tan(np.radians(30.0)) / V0)

print(f"Heading: reaches 90 deg +-2 deg after {t_settle:.1f} s, "
      f"overshoot {psi.max() - 90.0:.1f} deg")
print(f"Bank: max {phi.max():.1f} deg   "
      f"(30 deg limit -> coordinated turn rate {turn_rate:.2f} deg/s)")
print(f"Sideslip: max {np.degrees(np.abs(r.beta)).max():.2f} deg")
print(f"Altitude: max deviation {np.abs(r['h'] - ALT).max():.1f} m")
print(f"Speed: max deviation {np.abs(r.V - V0).max():.2f} m/s")
print(f"Controls: aileron {np.degrees(np.abs(r.da)).max():.1f} deg, "
      f"rudder {np.degrees(np.abs(r.dr)).max():.1f} deg, "
      f"throttle {r.throttle_cmd.min():.2f}-{r.throttle_cmd.max():.2f}")

fig = plt.figure(figsize=(13, 9))
ax0 = fig.add_subplot(1, 2, 1)
ax0.plot(r["y"] / 1000, r["x"] / 1000)
ax0.set_xlabel("East [km]")
ax0.set_ylabel("North [km]")
ax0.set_title("Ground track")
ax0.set_aspect("equal")
ax0.grid(alpha=0.3)

axs = [fig.add_subplot(4, 2, k) for k in (2, 4, 6, 8)]
axs[0].plot(r.t, psi, label="heading")
axs[0].axhline(90.0, color="k", ls="--", lw=1)
axs[0].set_ylabel("heading [deg]")
axs[1].plot(r.t, phi, label="bank")
axs[1].plot(r.t, np.degrees(r.phi_cmd), "--", lw=1, label="bank command")
axs[1].set_ylabel("bank [deg]")
axs[1].legend(fontsize=8)
axs[2].plot(r.t, np.degrees(r.beta))
axs[2].set_ylabel("sideslip [deg]")
axs[3].plot(r.t, r["h"] - ALT)
axs[3].set_ylabel("h error [m]")
axs[3].set_xlabel("time [s]")
for ax in axs:
    ax.grid(alpha=0.3)
fig.suptitle(f"90° heading change, {ALT / 1000:.0f} km, Mach {MACH} "
             f"(autothrottle + altitude hold + heading hold)")
fig.tight_layout()
fig.savefig(RESULTS / "sim_heading_turn.png", dpi=150)
print(f"Saved {RESULTS / 'sim_heading_turn.png'}")