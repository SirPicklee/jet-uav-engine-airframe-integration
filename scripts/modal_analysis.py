"""
Longitudinal modes from the linearized model at a trim point.

    python scripts/modal_analysis.py
"""

import numpy as np

from jet_uav_sim.dynamics import LongitudinalModel, state_from_trim
from jet_uav_sim.linearize import linearize, modes
from jet_uav_sim.trim import trim_level

ALT, MACH, FUEL = 6000.0, 0.6, 1800.0

model = LongitudinalModel()
trim = trim_level(ALT, MACH, model.mass(FUEL), aero=model.aero, deck=model.deck)
A, B = linearize(model, state_from_trim(trim, FUEL), np.radians(trim.de_deg), trim.throttle)
osc, real = modes(A)

print(f"Flight condition: {ALT:.0f} m, Mach {MACH}, mass {model.mass(FUEL):.0f} kg\n")
print("Oscillatory modes")
for o in osc:
    print(f"  wn = {o['wn']:.4f} rad/s   zeta = {o['zeta']:.3f}   period = {o['period']:.1f} s")
print("Real modes")
for r in real:
    print(f"  lambda = {r['eigenvalue']:.4f} 1/s   time constant = {r['time_constant']:.1f} s")