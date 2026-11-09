"""
Generate the thrust deck and its plots.

    python scripts/build_thrust_deck.py          # full grid (takes a few minutes)
    python scripts/build_thrust_deck.py --quick  # coarse grid for testing

Outputs:
    data/thrust_deck.csv
    results/thrust_deck_max_dry.png
    results/thrust_deck_tsfc.png
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.engine import make_engine
from jet_uav_sim.thrust_deck import generate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="coarse grid")
args = parser.parse_args()

if args.quick:
    altitudes = np.array([0.0, 6000.0, 12000.0])
    machs = np.array([0.0, 0.45, 0.9])
    throttles = np.array([0.0, 0.5, 1.0])
else:
    altitudes = np.arange(0.0, 12001.0, 2000.0)
    machs = np.round(np.arange(0.0, 0.91, 0.15), 2)
    throttles = np.round(np.linspace(0.0, 1.0, 6), 2)

design, _ = make_engine()
print(f"Design point (SLS): Fn = {design.net_thrust / 1e3:.1f} kN, "
      f"TSFC = {design.tsfc:.1f} g/(kN s), OPR = {design.opr:.1f}")
n = len(altitudes) * len(machs) * len(throttles)
print(f"Generating {n} points...")

t0 = time.time()
df = generate(altitudes, machs, throttles)
print(f"Done in {time.time() - t0:.0f} s, unconverged points: {(df.limiter == 'no_conv').sum()}")

DATA.mkdir(exist_ok=True)
df.to_csv(DATA / "thrust_deck.csv", index=False)
print(f"Saved {DATA / 'thrust_deck.csv'}")

# Plots -----------------------------------------------------------------------
RESULTS.mkdir(exist_ok=True)
mx = df[df.throttle == 1.0]

fig, ax = plt.subplots(figsize=(8, 5))
for alt in altitudes:
    s = mx[mx.altitude_m == alt]
    ax.plot(s.mach, s.thrust_N / 1e3, "o-", label=f"{alt / 1000:.0f} km")
ax.set_xlabel("Mach number")
ax.set_ylabel("Net thrust [kN]")
ax.set_title("Max dry thrust lapse")
ax.grid(alpha=0.3)
ax.legend(title="Altitude")
fig.tight_layout()
fig.savefig(RESULTS / "thrust_deck_max_dry.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for alt in altitudes:
    s = df[(df.altitude_m == alt) & (df.mach == machs[len(machs) // 2])]
    tsfc = s.fuel_flow_kg_s / s.thrust_N * 1e6
    ax.plot(s.thrust_N / 1e3, tsfc, "o-", label=f"{alt / 1000:.0f} km")
ax.set_xlabel("Net thrust [kN]")
ax.set_ylabel("TSFC [g/(kN s)]")
ax.set_title(f"Part-power TSFC at Mach {machs[len(machs) // 2]:.2f}")
ax.grid(alpha=0.3)
ax.legend(title="Altitude")
fig.tight_layout()
fig.savefig(RESULTS / "thrust_deck_tsfc.png", dpi=150)
plt.close(fig)
print(f"Plots saved in {RESULTS}")
