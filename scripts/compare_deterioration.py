"""
New vs. worn engine: radius vs. time on station.

Run radius_endurance.py with and without --worn first.

    python scripts/compare_deterioration.py

Output: results/deterioration_mission_impact.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
new = pd.read_csv(ROOT / "data" / "radius_endurance.csv")
worn = pd.read_csv(ROOT / "data" / "radius_endurance_worn.csv")


def max_radius(df):
    """Radius where time on station reaches zero (linear extrapolation of the last two points)."""
    r, t = df.radius_km.to_numpy(), df.loiter_h.to_numpy()
    slope = (t[-1] - t[-2]) / (r[-1] - r[-2])
    return r[-1] - t[-1] / slope


r_new, r_worn = max_radius(new), max_radius(worn)
d = new.merge(worn, on="radius_km", suffixes=("_new", "_worn"))
d["loss_min"] = 60 * (d.loiter_h_new - d.loiter_h_worn)
d["loss_pct"] = 100 * (1 - d.loiter_h_worn / d.loiter_h_new)

print(f"Max radius: new {r_new:.0f} km, worn {r_worn:.0f} km, loss {r_new - r_worn:.0f} km")
print(f"{'radius km':>10} {'loss min':>9} {'loss %':>7}")
for _, row in d.iterrows():
    print(f"{row.radius_km:10.0f} {row.loss_min:9.1f} {row.loss_pct:7.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.append(new.radius_km, r_new), np.append(new.loiter_h, 0), "o-", lw=2,
        label="new engine")
ax.plot(np.append(worn.radius_km, r_worn), np.append(worn.loiter_h, 0), "s-", lw=2,
        label="worn engine")
ax.annotate(f"−{r_new - r_worn:.0f} km", xy=(r_worn, 0.05), xytext=(r_worn - 250, 0.6),
            arrowprops={"arrowstyle": "->"})
ax.set_xlabel("Mission radius [km]")
ax.set_ylabel("Time on station [h]")
ax.set_title("Effect of engine deterioration on mission capability")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "results" / "deterioration_mission_impact.png", dpi=150)
print(f"Saved {ROOT / 'results' / 'deterioration_mission_impact.png'}")