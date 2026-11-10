"""
Mission radius vs. maximum time on station (the classic UAV capability chart).

For each radius: take-off, climb, cruise out, loiter until only the fuel for
the way home + reserve is left, cruise back, descend, land.
The fuel for the way home depends on the mass after the loiter, so it is
found by iteration (converges in 2-3 passes).

    python scripts/radius_endurance.py

Output: results/radius_endurance.png, data/radius_endurance.csv
"""

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jet_uav_sim.airframe import UAV
from jet_uav_sim.mission import Mission
from jet_uav_sim.thrust_deck import ThrustDeck
from jet_uav_sim.trim import DECK_PATH

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--worn", action="store_true", help="use the deteriorated engine deck")
args = parser.parse_args()
SUFFIX = "_worn" if args.worn else ""
DECK = ThrustDeck.from_csv(DECK_PATH.with_name(f"thrust_deck{SUFFIX}.csv"))

FUEL = UAV.mass.fuel_max
RESERVE = 0.10 * FUEL
LANDING_FUEL = 30.0
CRUISE_ALT, CRUISE_MACH = 10000.0, 0.75
LOITER_MACH = 0.475          # best loiter Mach at 10 km (find_loiter_optimum.py)
DT_LOITER = 300.0            # s, coarse step is enough for the loiter


def outbound(radius):
    m = Mission(mass0=UAV.mass.mtow, fuel0=FUEL, deck=DECK)
    m.fixed_fuel("taxi + take-off", fuel=60.0, duration_s=300.0)
    m.climb(CRUISE_ALT, mach=0.6)
    m.cruise(CRUISE_MACH, distance_m=radius - m.x, name="cruise out")
    return m


def fly_home(m, radius):
    """Return leg on a copy; returns the fuel it needs."""
    home = copy.deepcopy(m)
    f0 = home.fuel_used
    home.cruise(CRUISE_MACH, distance_m=radius - 30e3, name="cruise back")  # descent covers ~30 km
    home.descend(0.0, mach=0.5)
    home.fixed_fuel("landing", fuel=LANDING_FUEL)
    return home.fuel_used - f0, home


def max_loiter(radius):
    m_out = outbound(radius)
    home_fuel, _ = fly_home(m_out, radius)          # first guess: mass before loiter
    for _ in range(3):
        m = copy.deepcopy(m_out)
        budget = FUEL - RESERVE - home_fuel
        if budget <= m.fuel_used:
            return None                               # cannot even turn around
        t0 = m.t
        m.cruise(LOITER_MACH, fuel_stop=budget, dt=DT_LOITER, name="loiter")
        new_home, home = fly_home(m, radius)
        if abs(new_home - home_fuel) < 1.0:
            break
        home_fuel = new_home
    return (m.t - t0) / 3600, home.t / 3600, FUEL - home.fuel_used


rows = []
for radius_km in range(100, 1301, 100):
    try:
        res = max_loiter(radius_km * 1e3)
    except RuntimeError as err:
        print(f"{radius_km:5d} km   failed: {err}")
        break
    if res is None:
        print(f"{radius_km:5d} km   not reachable with reserve")
        break
    loiter_h, total_h, left = res
    rows.append({"radius_km": radius_km, "loiter_h": loiter_h, "mission_h": total_h,
                 "fuel_left_kg": left})
    print(f"{radius_km:5d} km   loiter {loiter_h:5.2f} h   mission {total_h:5.2f} h   "
          f"fuel left {left:5.0f} kg")

df = pd.DataFrame(rows)
(ROOT / "data").mkdir(exist_ok=True)
df.to_csv(ROOT / "data" / f"radius_endurance{SUFFIX}.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df.radius_km, df.loiter_h, "o-", lw=2)
ax.set_xlabel("Mission radius [km]")
ax.set_ylabel("Time on station [h]")
ax.set_title(f"Radius vs. time on station (cruise M{CRUISE_MACH} / loiter M{LOITER_MACH} "
             f"at {CRUISE_ALT / 1000:.0f} km, {RESERVE:.0f} kg reserve)")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / f"radius_endurance{SUFFIX}.png", dpi=150)
print(f"Saved {RESULTS / f'radius_endurance{SUFFIX}.png'}")