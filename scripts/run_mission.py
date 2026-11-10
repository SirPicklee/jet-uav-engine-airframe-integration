"""
Example ISR (reconnaissance) mission: transit out, loiter on station, transit back.

    python scripts/run_mission.py

Also checks the outbound cruise against the Breguet range equation.
Outputs:
    data/mission_log.csv
    results/mission_profile.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jet_uav_sim.airframe import UAV
from jet_uav_sim.atmosphere import G0, isa
from jet_uav_sim.mission import Mission
from jet_uav_sim.trim import trim_level

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FUEL = UAV.mass.fuel_max
RESERVE = 0.10 * FUEL
CRUISE_ALT, CRUISE_MACH = 10000.0, 0.75
LOITER_MACH = 0.55
RADIUS = 400e3        # m
LOITER = 2 * 3600.0   # s

m = Mission(mass0=UAV.mass.mtow, fuel0=FUEL)
m.fixed_fuel("taxi + take-off", fuel=60.0, duration_s=300.0)
m.climb(CRUISE_ALT, mach=0.6)

# Breguet check on the outbound cruise: constant altitude and Mach
x_climb = m.x
cruise_dist = RADIUS - x_climb
trim0 = trim_level(m.h, CRUISE_MACH, m.mass, aero=m.aero, deck=m.deck)
V = CRUISE_MACH * float(isa(m.h)[3])
c = trim0.fuel_flow / trim0.thrust                      # kg/(N s)
mass_in = m.mass
m.cruise(CRUISE_MACH, distance_m=cruise_dist, name="cruise out")
fuel_sim = mass_in - m.mass
fuel_breguet = mass_in * (1 - np.exp(-cruise_dist * c * G0 / (V * trim0.lift_to_drag)))

m.cruise(LOITER_MACH, duration_s=LOITER, name="loiter")
m.cruise(CRUISE_MACH, distance_m=cruise_dist, name="cruise back")
m.descend(0.0, mach=0.5)
m.fixed_fuel("landing", fuel=30.0, duration_s=300.0)

log = m.log()
(ROOT / "data").mkdir(exist_ok=True)
log.to_csv(ROOT / "data" / "mission_log.csv", index=False)

# Summary per segment ----------------------------------------------------------------
print(f"{'segment':>16} {'time min':>9} {'fuel kg':>8}")
prev_t, prev_f = 0.0, 0.0
for seg in log.segment.unique()[1:]:
    last = log[log.segment == seg].iloc[-1]
    print(f"{seg:>16} {(last.t_s - prev_t) / 60:9.1f} {last.fuel_used_kg - prev_f:8.1f}")
    prev_t, prev_f = last.t_s, last.fuel_used_kg

left = FUEL - m.fuel_used
print(f"\nMission time {m.t / 3600:.2f} h, fuel used {m.fuel_used:.0f} kg, "
      f"fuel left {left:.0f} kg (reserve {RESERVE:.0f} kg) -> "
      f"{'OK' if left >= RESERVE else 'NOT FEASIBLE'}")
print(f"Breguet check, outbound cruise: simulation {fuel_sim:.1f} kg, "
      f"Breguet {fuel_breguet:.1f} kg ({100 * (fuel_sim / fuel_breguet - 1):+.1f} %)")

# Plot ------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for seg in log.segment.unique():
    s = log[log.segment == seg]
    axs[0].plot(s.t_s / 60, s.h_m / 1000, "-", lw=2, label=seg)
    axs[1].plot(s.t_s / 60, FUEL - s.fuel_used_kg, "-", lw=2)
axs[1].axhline(RESERVE, color="r", ls="--", lw=1, label="reserve")
axs[0].set_ylabel("Altitude [km]")
axs[1].set_ylabel("Fuel remaining [kg]")
axs[1].set_xlabel("Time [min]")
axs[0].legend(fontsize=8)
axs[1].legend(fontsize=8)
for ax in axs:
    ax.grid(alpha=0.3)
axs[0].set_title(f"ISR mission: {RADIUS / 1e3:.0f} km radius, "
                 f"{LOITER / 3600:.0f} h loiter")
fig.tight_layout()
fig.savefig(RESULTS / "mission_profile.png", dpi=150)
print(f"Saved {RESULTS / 'mission_profile.png'}")