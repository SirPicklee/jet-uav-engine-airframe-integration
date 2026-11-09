"""
International Standard Atmosphere (ISA), troposphere and lower stratosphere (0-20 km).
"""

import numpy as np

G0 = 9.80665       # m/s^2
R_AIR = 287.053    # J/(kg K)
GAMMA = 1.4

T0, P0 = 288.15, 101325.0    # sea level
LAPSE = -0.0065              # K/m, troposphere
H_TROP = 11000.0             # m, tropopause
T_TROP = T0 + LAPSE * H_TROP
P_TROP = P0 * (T_TROP / T0) ** (-G0 / (LAPSE * R_AIR))


def isa(altitude):
    """Temperature [K], pressure [Pa], density [kg/m^3] and speed of sound [m/s]."""
    h = np.asarray(altitude, dtype=float)
    T = np.where(h <= H_TROP, T0 + LAPSE * h, T_TROP)
    p = np.where(
        h <= H_TROP,
        P0 * (T / T0) ** (-G0 / (LAPSE * R_AIR)),
        P_TROP * np.exp(-G0 * (h - H_TROP) / (R_AIR * T_TROP)),
    )
    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA * R_AIR * T)
    return T, p, rho, a