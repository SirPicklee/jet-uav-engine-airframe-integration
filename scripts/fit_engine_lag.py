"""
Engine acceleration (idle -> max dry, sea level static) with turbofan-cycle's
transient model, and a first-order lag fitted to the thrust response.

The fitted time constant replaces the guessed tau_engine in the flight model.

    python scripts/fit_engine_lag.py

Output: results/engine_lag_fit.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from jet_uav_sim.engine import T4_IDLE, UAV_ENGINE, make_engine
from turbofan.transient import Spools, accelerate, steady_schedule

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Rotor inertias and speeds scaled from the CFM56-class defaults by mass flow:
#   radius ~ W^0.5  ->  rpm ~ W^-0.5,  J ~ m r^2 ~ W^2   (rough similarity scaling)
ref = Spools()
k = UAV_ENGINE.W / 355.0
spools = Spools(J_L=ref.J_L * k**2, J_H=ref.J_H * k**2,
                rpm_L=ref.rpm_L / np.sqrt(k), rpm_H=ref.rpm_H / np.sqrt(k))
print(f"Spools: J_L {spools.J_L:.3f} kg m2, J_H {spools.J_H:.4f} kg m2, "
      f"N1 {spools.rpm_L:.0f} rpm, N2 {spools.rpm_H:.0f} rpm")

design, eng = make_engine()
Wf_max = design.fuel_flow
print(f"Target: max dry, Wf = {Wf_max:.3f} kg/s, Fn = {design.net_thrust / 1e3:.1f} kN")

print("Building acceleration fuel schedule ...")
schedule = steady_schedule(eng, margin=0.30,
                           T4_values=np.arange(UAV_ENGINE.T4, T4_IDLE - 1, -50.0))

print("Simulating throttle slam ...")
r = accelerate(eng, Wf_max, schedule, T4_idle=T4_IDLE, t_end=10.0, dt=0.05,
               spools=spools, name="UAV engine")
if r.surge_time is not None:
    print(f"WARNING: HPC surge at t = {r.surge_time:.2f} s")
if r.failed_time is not None:
    print(f"WARNING: solver failed at t = {r.failed_time:.2f} s")

t = np.array(r.t)
F = np.array(r.thrust)
F0, F_end = F[0], design.net_thrust
f_norm = (F - F0) / (F_end - F0)


def first_order(t, tau):
    return 1.0 - np.exp(-t / tau)


(tau,), _ = curve_fit(first_order, t, f_norm, p0=[1.0])
t95 = r.time_to(F0 + 0.95 * (F_end - F0))
print(f"\nFitted first-order lag: tau = {tau:.2f} s")
print(f"Time to 95 % thrust: {t95:.2f} s" if t95 else "95 % thrust not reached")
print(f"Min HPC surge margin: {100 * min(r.sm_hpc):.1f} %, max T4: {max(r.T4):.0f} K")

fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
axs[0].plot(t, F / 1e3, label="turbofan-cycle transient")
axs[0].plot(t, (F0 + (F_end - F0) * first_order(t, tau)) / 1e3, "--",
            label=f"first-order fit, tau = {tau:.2f} s")
axs[0].set_ylabel("Net thrust [kN]")
axs[0].legend()
axs[1].plot(t, 100 * np.array(r.sm_hpc))
axs[1].axhline(0, color="r", lw=1)
axs[1].set_ylabel("HPC surge margin [%]")
axs[1].set_xlabel("time [s]")
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].set_title("Idle to max dry acceleration, sea level static")
fig.tight_layout()
fig.savefig(RESULTS / "engine_lag_fit.png", dpi=150)
print(f"Saved {RESULTS / 'engine_lag_fit.png'}")