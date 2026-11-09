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
from turbofan.offdesign_maps import build_engine, off_design_maps  # noqa: E402

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


def make_engine():
    """Design-point result and the map-based off-design engine."""
    design = run_design_point(UAV_ENGINE)
    return design, build_engine(design)


def max_point(eng, altitude, mach, guess=None):
    """Highest-thrust point allowed by MAX_DRY. Returns (MapPoint, limiter, x)."""
    return max_rating(eng, altitude, mach, MAX_DRY, guess=guess)


def part_power_point(eng, altitude, mach, T4, guess=None):
    """Operating point at a given T4. Returns (MapPoint, x)."""
    return off_design_maps(eng, altitude, mach, T4, guess=guess)
