"""
Engine model for the jet UAV: a small low-bypass turbofan built with turbofan-cycle.

The engine is sized for a ~6 t jet UAV (about 29 kN sea-level static, dry).
turbofan-cycle is used as a git submodule in external/turbofan-cycle.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "turbofan-cycle"))

from turbofan.cycle import TurbofanDesign, run_design_point  # noqa: E402
from turbofan.limits import Rating, max_rating  # noqa: E402
from turbofan.offdesign_maps import (  # noqa: E402
    Health, build_engine, off_design_maps, with_health,
)

# Design point: sea level static, ISA
UAV_ENGINE = TurbofanDesign(
    W=45.0,        # kg/s
    bpr=0.9,       # low bypass, fighter-class
    fan_pr=2.2,
    lpc_pr=1.6,
    hpc_pr=6.0,    # OPR ~21
    T4=1650.0,     # K
)

# Maximum continuous dry rating (no afterburner)
MAX_DRY = Rating("Max dry", T4_max=1650.0, N_L_max=1.05, N_H_max=1.05,
                 n_fan_max=1.08, T3_max=800.0)

T4_IDLE = 850.0    # K, lowest T4 in the deck (~10 % of max thrust at SLS)



# Representative mid-life wear (illustrative, not fleet data): compressors foul,
# turbine tip clearances open. Signs follow turbofan-cycle's deterioration module.
WORN = Health(fan_eff=-0.01, hpc_eff=-0.02, hpc_flow=-0.01,
              hpt_eff=-0.015, hpt_flow=+0.01, lpt_eff=-0.01)


def make_engine(health=None):
    """Design-point result and the map-based off-design engine (optionally worn)."""
    design = run_design_point(UAV_ENGINE)
    eng = build_engine(design)
    if health is not None:
        eng = with_health(eng, health)
    return design, eng


def max_point(eng, altitude, mach, guess=None):
    """Highest-thrust point allowed by MAX_DRY. Returns (MapPoint, limiter, x)."""
    return max_rating(eng, altitude, mach, MAX_DRY, guess=guess)


def part_power_point(eng, altitude, mach, T4, guess=None):
    """Operating point at a given T4. Returns (MapPoint, x)."""
    return off_design_maps(eng, altitude, mach, T4, guess=guess)
