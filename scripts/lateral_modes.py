"""
Lateral-directional modes of the 6-DoF model at a trim point, compared with
the classical approximations (Nelson, Flight Stability and Automatic Control)
and the MIL-F-8785C level 1 limits (class IV, category B).

    python scripts/lateral_modes.py
"""

import numpy as np

from jet_uav_sim.atmosphere import isa
from jet_uav_sim.dynamics6dof import SixDofModel, state6_from_trim
from jet_uav_sim.linearize import linearize_lateral, modes
from jet_uav_sim.trim import trim_level

ALT, MACH, FUEL = 6000.0, 0.6, 1800.0

model = SixDofModel()
mass = model.mass(FUEL)
trim = trim_level(ALT, MACH, mass, aero=model.aero, deck=model.deck)
s0 = state6_from_trim(trim, FUEL)
A, B = linearize_lateral(model, s0, (0.0, np.radians(trim.de_deg), 0.0, trim.throttle))
osc, real = modes(A)

dutch = osc[0]
roll = max(real, key=lambda m: abs(m["eigenvalue"]))
spiral = min(real, key=lambda m: abs(m["eigenvalue"]))

# Dimensional derivatives for the classical approximations
af, lat = model.aero.airframe, model.lateral
S, b = af.wing.area, af.wing.span
_, _, rho, a = isa(ALT)
V = MACH * float(a)
q_bar = 0.5 * float(rho) * V**2
Ixx, _, Izz = af.inertia(mass)
k = b / (2 * V)
Yb = q_bar * S * lat.cy_beta / mass
Lb = q_bar * S * b * lat.cl_beta / Ixx
Lp = q_bar * S * b * lat.cl_p * k / Ixx
Lr = q_bar * S * b * lat.cl_r * k / Ixx
Nb = q_bar * S * b * lat.cn_beta / Izz
Nr = q_bar * S * b * lat.cn_r * k / Izz
Np = q_bar * S * b * lat.cn_p * k / Izz

wn_dr = np.sqrt(Nb + Yb * Nr / V)                 # Dutch roll, 2-DoF (sideslip + yaw)
zeta_dr = -(Yb / V + Nr) / (2 * wn_dr)
tau_roll = -1.0 / Lp                              # pure rolling
lam_spiral = 9.80665 / V * (Lb * Nr - Nb * Lr) / (Lp * Nb - Lb * Np)   # spiral (Cook)

print(f"Flight condition: {ALT:.0f} m, Mach {MACH}, mass {mass:.0f} kg\n")
print(f"{'':>22} {'6-DoF linear':>14} {'approximation':>14}")
print(f"{'Dutch roll wn [rad/s]':>22} {dutch['wn']:14.3f} {wn_dr:14.3f}")
print(f"{'Dutch roll zeta':>22} {dutch['zeta']:14.3f} {zeta_dr:14.3f}")
print(f"{'Dutch roll period [s]':>22} {dutch['period']:14.2f}")
print(f"{'Roll tau [s]':>22} {roll['time_constant']:14.3f} {tau_roll:14.3f}")
print(f"{'Spiral lambda [1/s]':>22} {spiral['eigenvalue']:14.4f} {lam_spiral:14.4f}")

print("\nMIL-F-8785C, class IV, category B, level 1:")
checks = [
    ("Dutch roll zeta >= 0.08", dutch["zeta"] >= 0.08),
    ("Dutch roll zeta*wn >= 0.15 rad/s", dutch["zeta"] * dutch["wn"] >= 0.15),
    ("Dutch roll wn >= 0.4 rad/s", dutch["wn"] >= 0.4),
    ("Roll mode tau <= 1.4 s", roll["time_constant"] <= 1.4),
]
if spiral["eigenvalue"] > 0:
    t2 = np.log(2) / spiral["eigenvalue"]
    checks.append((f"Spiral time to double >= 20 s ({t2:.0f} s)", t2 >= 20))
else:
    checks.append(("Spiral stable", True))
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")