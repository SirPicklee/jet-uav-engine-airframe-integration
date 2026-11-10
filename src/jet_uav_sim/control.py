"""
Longitudinal autopilot: autothrottle and altitude hold.

  Autothrottle  : PI on airspeed error -> throttle command (trim feedforward,
                  anti-windup by conditional integration)
  Altitude hold : outer loop  h error, climb rate -> pitch attitude command
                  inner loop  pitch attitude, pitch rate -> elevator

Sign convention: positive elevator (trailing edge down) pitches the nose down
(Cm_de < 0), so a nose-up command needs negative elevator.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PI:
    kp: float
    ki: float
    out_min: float
    out_max: float
    feedforward: float = 0.0
    integral: float = 0.0

    def update(self, error, dt):
        u = self.feedforward + self.kp * error + self.ki * self.integral
        saturated_high = u >= self.out_max and error > 0
        saturated_low = u <= self.out_min and error < 0
        if not (saturated_high or saturated_low):       # anti-windup
            self.integral += error * dt
        return float(np.clip(u, self.out_min, self.out_max))


@dataclass
class Autothrottle:
    throttle_trim: float
    kp: float = 0.05           # throttle per m/s
    ki: float = 0.005          # throttle per (m/s * s)
    pi: PI = field(init=False)

    def __post_init__(self):
        self.pi = PI(self.kp, self.ki, 0.0, 1.0, feedforward=self.throttle_trim)

    def command(self, V_cmd, V, dt):
        return self.pi.update(V_cmd - V, dt)


@dataclass
class AltitudeHold:
    theta_trim: float           # rad
    de_trim: float              # rad
    k_h: float = 0.0015         # rad of pitch per m of altitude error
    k_h_i: float = 0.00003      # rad of pitch per (m * s), removes steady-state error
    k_hdot: float = 0.005       # rad of pitch per m/s of climb rate
    k_theta: float = 2.0        # rad of elevator per rad of pitch error
    k_q: float = 0.5            # rad of elevator per rad/s of pitch rate
    theta_limit: float = np.radians(10.0)
    theta_rate_limit: float = np.radians(3.0)   # rad/s, smooths large steps
    de_limit: float = np.radians(25.0)
    integral: float = 0.0
    theta_cmd_prev: float = None

    def command(self, h_cmd, h, h_dot, theta, q, dt):
        err = h_cmd - h
        lo = self.theta_trim - self.theta_limit
        hi = self.theta_trim + self.theta_limit
        raw = (self.theta_trim + self.k_h * err + self.k_h_i * self.integral
               - self.k_hdot * h_dot)
        if lo < raw < hi:                       # anti-windup
            self.integral += err * dt
        target = np.clip(raw, lo, hi)

        prev = self.theta_trim if self.theta_cmd_prev is None else self.theta_cmd_prev
        step = self.theta_rate_limit * dt
        theta_cmd = prev + np.clip(target - prev, -step, step)
        self.theta_cmd_prev = theta_cmd

        de = self.de_trim + self.k_theta * (theta - theta_cmd) + self.k_q * q
        return float(np.clip(de, -self.de_limit, self.de_limit)), float(theta_cmd)

    

def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class HeadingHold:
    """Heading hold through bank, with a yaw damper and turn coordination.

    Outer loop : heading error -> bank command (limited)
    Inner loop : bank error, roll rate -> aileron
    Rudder     : yaw rate minus the coordinated-turn yaw rate (yaw damper),
                 plus sideslip feedback

    Signs: positive aileron rolls right; positive rudder yaws the nose left.
    """
    k_psi: float = 2.0          # rad of bank per rad of heading error
    k_phi: float = 0.2          # rad of aileron per rad of bank error
    k_p: float = 0.03           # rad of aileron per rad/s of roll rate
    k_r: float = 0.4            # rad of rudder per rad/s of yaw rate error
    k_beta: float = 0.3         # rad of rudder per rad of sideslip
    phi_limit: float = np.radians(30.0)
    da_limit: float = np.radians(20.0)
    dr_limit: float = np.radians(25.0)

    def command(self, psi_cmd, psi, phi, theta, p, r, beta, V):
        phi_cmd = np.clip(self.k_psi * wrap_angle(psi_cmd - psi),
                          -self.phi_limit, self.phi_limit)
        da = self.k_phi * (phi_cmd - phi) - self.k_p * p
        r_coord = 9.80665 / V * np.sin(phi) * np.cos(theta)
        dr = self.k_r * (r - r_coord) - self.k_beta * beta
        return (float(np.clip(da, -self.da_limit, self.da_limit)),
                float(np.clip(dr, -self.dr_limit, self.dr_limit)),
                float(phi_cmd))