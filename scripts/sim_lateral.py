"""
6-DoF time simulations of the lateral-directional modes.

  Rudder doublet: Dutch roll; period and damping measured from the yaw-rate
                  peaks and compared with the linear model (2.02 s, zeta 0.104)
  Aileron pulse : roll to ~20 deg bank and let go; the bank decays through the
                  spiral mode (linear model: time constant ~68 s)

Elevator and throttle stay at their trim values (stick-fixed response).

    python scripts/sim_lateral.py

Outputs: results/sim_rudder_doublet.png, results/sim_aileron_pulse.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.dynamics6dof import STATE_NAMES_6DOF, SixDofModel, state6_from_trim
from jet_uav_sim.sim import rk4_step
from jet_uav_sim.trim import trim_level

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ALT, MACH, FUEL = 6000.0, 0.6, 1800.0
model = SixDofModel()
trim = trim_level(ALT, MACH, model.mass(FUEL), aero=model.aero, deck=model.deck)
de0, thr0 = np.radians(trim.de_deg), trim.throttle
col = {name: i for i, name in enumerate(STATE_NAMES_6DOF)}


def run(controls, t_end, dt):
    """controls(t) -> (aileron, rudder) [rad]. Returns time and state history."""
    s = state6_from_trim(trim, FUEL)
    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    out = np.zeros((n, len(s)))
    for i in range(n):
        out[i] = s
        if i < n - 1:
            da, dr = controls(t[i])
            s = rk4_step(lambda st: model.derivatives(st, da, de0, dr, thr0), s, dt)
    return t, out


def peaks(t, y):
    i = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return t[i], y[i]


def sideslip_deg(out):
    V = np.linalg.norm(out[:, [col["u"], col["v"], col["w"]]], axis=1)
    return np.degrees(np.arcsin(out[:, col["v"]] / V))


# 1) Rudder doublet -------------------------------------------------------------------
DR = np.radians(3.0)


def rudder_doublet(t):
    if 1.0 <= t < 2.0:
        return 0.0, DR
    if 2.0 <= t < 3.0:
        return 0.0, -DR
    return 0.0, 0.0


t, out = run(rudder_doublet, 20.0, 0.01)
r = out[:, col["r"]]
tp, rp = peaks(t[t > 3.5], r[t > 3.5])
tp, rp = tp[rp > 0][:4], rp[rp > 0][:4]
period = np.mean(np.diff(tp))
delta = np.mean(np.log(rp[:-1] / rp[1:]))
zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)
print(f"Dutch roll from simulation: period {period:.2f} s, zeta {zeta:.3f}   "
      f"(linear model: 2.02 s, 0.104)")

fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
axs[0].plot(t, sideslip_deg(out))
axs[0].set_ylabel("sideslip [deg]")
axs[1].plot(t, np.degrees(r))
axs[1].plot(tp, np.degrees(rp), "ko", ms=4, label="peaks used")
axs[1].set_ylabel("yaw rate [deg/s]")
axs[1].legend(fontsize=8)
axs[2].plot(t, np.degrees(out[:, col["p"]]))
axs[2].set_ylabel("roll rate [deg/s]")
axs[3].plot(t, np.degrees(out[:, col["phi"]]))
axs[3].set_ylabel("bank [deg]")
axs[3].set_xlabel("time [s]")
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].set_title(f"Rudder doublet ±3°: Dutch roll (period {period:.2f} s, zeta {zeta:.3f})")
fig.tight_layout()
fig.savefig(RESULTS / "sim_rudder_doublet.png", dpi=150)
plt.close(fig)

# 2) Aileron pulse --------------------------------------------------------------------
DA = np.radians(5.0)
t, out = run(lambda t: (DA if 1.0 <= t < 1.5 else 0.0, 0.0), 150.0, 0.02)
phi = np.degrees(out[:, col["phi"]])
print(f"Peak bank after the pulse: {phi.max():.1f} deg")
fit = (t > 15.0) & (phi > 1.0)
slope = np.polyfit(t[fit], np.log(phi[fit]), 1)[0]
print(f"Bank decay from simulation: time constant {-1 / slope:.0f} s   "
      f"(linear spiral mode: 68 s)")
print(f"Altitude lost: {ALT - out[:, col['h']].min():.0f} m, "
      f"heading change: {np.degrees(out[-1, col['psi']]):.0f} deg")

fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
axs[0].plot(t, phi)
axs[0].set_ylabel("bank [deg]")
axs[1].plot(t, np.degrees(out[:, col["psi"]]))
axs[1].set_ylabel("heading [deg]")
axs[2].plot(t, out[:, col["h"]])
axs[2].set_ylabel("h [m]")
axs[2].set_xlabel("time [s]")
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].set_title("Aileron pulse 5° for 0.5 s, then hands off: spiral mode")
fig.tight_layout()
fig.savefig(RESULTS / "sim_aileron_pulse.png", dpi=150)
print(f"Saved plots in {RESULTS}")