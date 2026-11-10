"""
Monte Carlo uncertainty on the maximum mission radius.

Uncertain inputs (sampled independently):
  oswald_factor  U(0.85, 1.00)  Raymer's swept-wing estimate is optimistic
  cfe_factor     U(0.85, 1.15)  conceptual zero-lift drag
  empty_dm       N(0, 3 %)      empty mass estimation error / weight growth [kg]
  wear           U(0, 1)        new engine (0) to the mid-life WORN case (1)

Max radius = radius at which a no-loiter out-and-back mission uses all the fuel
except the reserve. Fuel used is close to linear in radius, so two missions per
sample (900 and 1200 km) are enough to interpolate it.

    python scripts/monte_carlo.py -n 20    # quick run to time it
    python scripts/monte_carlo.py          # 200 samples

Outputs: data/monte_carlo.csv, results/monte_carlo.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jet_uav_sim.aero import AeroModel
from jet_uav_sim.airframe import UAV
from jet_uav_sim.mission import Mission
from jet_uav_sim.thrust_deck import BlendedDeck, ThrustDeck
from jet_uav_sim.trim import DECK_PATH

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=200, help="number of samples")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

FUEL = UAV.mass.fuel_max
RESERVE = 0.10 * FUEL
CRUISE_ALT, CRUISE_MACH = 10000.0, 0.75
R1, R2 = 900e3, 1200e3
NEW = ThrustDeck.from_csv(DECK_PATH)
WORN = ThrustDeck.from_csv(DECK_PATH.with_name("thrust_deck_worn.csv"))
CFE0 = AeroModel().cfe


def fuel_for_radius(radius, aero, deck, mass0):
    """Fuel for a no-loiter out-and-back mission (same segments as radius_endurance.py)."""
    m = Mission(mass0=mass0, fuel0=1e9, aero=aero, deck=deck)   # no out-of-fuel stop
    m.fixed_fuel("taxi + take-off", fuel=60.0, duration_s=300.0)
    m.climb(CRUISE_ALT, mach=0.6)
    m.cruise(CRUISE_MACH, distance_m=radius - m.x, dt=300.0, name="cruise out")
    m.cruise(CRUISE_MACH, distance_m=radius - 30e3, dt=300.0, name="cruise back")
    m.descend(0.0, mach=0.5)
    m.fixed_fuel("landing", fuel=30.0)
    return m.fuel_used


def max_radius(oswald_factor=1.0, cfe_factor=1.0, empty_dm=0.0, wear=0.0):
    """Maximum radius [km] for one set of inputs."""
    aero = AeroModel(oswald_factor=oswald_factor, cfe=CFE0 * cfe_factor)
    deck = BlendedDeck(NEW, WORN, wear)
    mass0 = UAV.mass.mtow + empty_dm
    f1 = fuel_for_radius(R1, aero, deck, mass0)
    f2 = fuel_for_radius(R2, aero, deck, mass0)
    usable = FUEL - RESERVE
    return (R1 + (usable - f1) * (R2 - R1) / (f2 - f1)) / 1e3


# 1) Design values and one-at-a-time sensitivity --------------------------------------
r_nom = max_radius()
print(f"Design values: max radius {r_nom:.0f} km")

dm = 0.06 * UAV.mass.empty
ranges = {"oswald_factor": (0.85, 1.00), "cfe_factor": (1.15, 0.85),
          "empty_dm": (dm, -dm), "wear": (1.0, 0.0)}          # (bad end, good end)
labels = {"oswald_factor": "Oswald factor x0.85 / x1.00",
          "cfe_factor": "CD0 x1.15 / x0.85",
          "empty_dm": f"Empty mass +{dm:.0f} / -{dm:.0f} kg",
          "wear": "Engine wear 1 / 0"}
tornado = []
for key, (bad, good) in ranges.items():
    r_bad, r_good = max_radius(**{key: bad}), max_radius(**{key: good})
    tornado.append((key, r_bad, r_good))
    print(f"  {labels[key]:>30}: {r_bad:6.0f} / {r_good:6.0f} km")
tornado.sort(key=lambda t: abs(t[2] - t[1]))

# 2) Monte Carlo ------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
n = args.n
samples = pd.DataFrame({
    "oswald_factor": rng.uniform(0.85, 1.00, n),
    "cfe_factor": rng.uniform(0.85, 1.15, n),
    "empty_dm": rng.normal(0.0, 0.03 * UAV.mass.empty, n),
    "wear": rng.uniform(0.0, 1.0, n),
})
radii = []
for k, row in samples.iterrows():
    try:
        radii.append(max_radius(**row.to_dict()))
    except RuntimeError:
        radii.append(np.nan)
    if (k + 1) % 10 == 0:
        print(f"  sample {k + 1}/{n}")
samples["max_radius_km"] = radii
(ROOT / "data").mkdir(exist_ok=True)
samples.to_csv(ROOT / "data" / "monte_carlo.csv", index=False)

r = samples.max_radius_km.dropna()
p5, p50, p95 = np.percentile(r, [5, 50, 95])
print(f"\nMonte Carlo, {len(r)} samples: mean {r.mean():.0f} km, std {r.std():.0f} km")
print(f"  P5 {p5:.0f} km   P50 {p50:.0f} km   P95 {p95:.0f} km")
print(f"  probability of reaching the design-value radius: {100 * (r >= r_nom).mean():.0f} %")
print("Spearman rank correlation with max radius:")
corr = samples.corr(method="spearman")["max_radius_km"].drop("max_radius_km")
for key, c in corr.items():
    print(f"  {key:>14} {c:+.2f}")

# 3) Plots ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(r, bins=25, color="tab:blue", alpha=0.75)
for p, ls, name in [(p5, ":", "P5"), (p50, "-", "P50"), (p95, ":", "P95")]:
    ax1.axvline(p, color="k", ls=ls, lw=1, label=f"{name} {p:.0f} km")
ax1.axvline(r_nom, color="r", ls="--", lw=1.5, label=f"design values {r_nom:.0f} km")
ax1.set_xlabel("Max mission radius [km]")
ax1.set_ylabel("Samples")
ax1.set_title(f"Monte Carlo, {len(r)} samples")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

for i, (key, r_bad, r_good) in enumerate(tornado):
    ax2.barh(i, r_bad - r_nom, left=r_nom, color="tab:red", alpha=0.8)
    ax2.barh(i, r_good - r_nom, left=r_nom, color="tab:green", alpha=0.8)
ax2.set_yticks(range(len(tornado)), [labels[t[0]] for t in tornado])
ax2.axvline(r_nom, color="k", lw=1)
ax2.set_xlabel("Max mission radius [km]")
ax2.set_title("One-at-a-time sensitivity (red: bad end, green: good end)")
ax2.grid(alpha=0.3, axis="x")

fig.tight_layout()
fig.savefig(RESULTS / "monte_carlo.png", dpi=150)
print(f"Saved {RESULTS / 'monte_carlo.png'}")