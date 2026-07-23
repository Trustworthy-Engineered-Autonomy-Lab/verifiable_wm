import numpy as np

from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.model import *
from pybdr.algorithm import ASB2008CDC
from pybdr.geometry.operation import boundary, cvt2

from StarV.set.star import Star
from StarV.dynamic.Sine import SinLayer

from sympy import Matrix, cos, sin

import dynamic

class Pendulum(dynamic.Pendulum):
    def __init__(self, *args, **kwargs):
        # Keep existing callers that pass lp_solver= compatible; Pendulum
        # reachability itself is intentionally fixed to Gurobi.
        kwargs.pop("lp_solver", None)
        super().__init__(*args, **kwargs)

    def angle_normalize(self, x):
        """Normalize angle to [-π, π] range"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def _validate_star_bounds(full_bound: np.ndarray):
        for index, name in enumerate(("theta", "omega", "action", "sin(theta)")):
            lower, upper = full_bound[:, index]
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError(
                    f"Invalid {name} bound: lower={lower}, upper={upper}"
                )

    def step(self, bound: np.ndarray):
        # Interval form of dynamic.Pendulum.step:
        #   torque = clip(u, -1, 1) * max_torque
        #   omega' = omega + (3g/(2l)*sin(theta) + 3/(m*l^2)*torque)*dt,
        #            clipped to [-max_speed, max_speed]
        #   theta' = theta + omega'*dt, normalized to [-pi, pi]
        # Compute sin(theta) using SinLayer
        theta_min, theta_max = bound[:,0]

        L_sin = SinLayer()
        lb_theta = np.array([theta_min], dtype=np.float32)
        ub_theta = np.array([theta_max], dtype=np.float32)
        S_theta = Star(lb_theta, ub_theta)

        IM_sin = L_sin.reach(
            S_theta, method='approx', lp_solver='gurobi', RF=0.0
        )
        try:
            z_bound = np.array(IM_sin.getRanges(lp_solver='gurobi'))
        except Exception as e:
            z_bound = np.array(IM_sin.getRanges(lp_solver='estimate'))

        full_bound = np.concatenate([bound, z_bound], axis = 1)
        # dynamic.Pendulum clamps the raw action to [-1, 1] before scaling
        # by max_torque; the interval version must match.
        full_bound[:,2] = np.clip(full_bound[:,2], -1.0, 1.0)
        self._validate_star_bounds(full_bound)
        S_full = Star(full_bound[0], full_bound[1])

        # Apply dynamics on [theta, omega, u, sin(theta)]. Coefficients are
        # derived from the rollout parameters instead of hard-coded so both
        # implementations always describe the same system. c_u multiplies
        # the raw controller output u, so it must include the max_torque
        # scaling that dynamic.Pendulum applies.
        c_sin = 3.0 * self.g / (2.0 * self.l) * self.dt
        c_u = 3.0 / (self.m * self.l ** 2) * self.max_torque * self.dt
        # The theta' row uses the un-clipped omega'; whenever omega' exceeds
        # max_speed this only widens theta', so the bound stays sound.
        M = np.array([[1.0, self.dt, self.dt * c_u, self.dt * c_sin],
                      [0.0, 1.0, c_u, c_sin]], dtype=np.float32)
        b_dyn = np.zeros(2, dtype=np.float32)
        S_next = S_full.affineMap(M, b_dyn)

        # Step 7: Get bounds BEFORE clipping
        next_bound = np.array(S_next.getRanges('gurobi', RF=0.0))

        # Clip omega' like the rollout dynamics
        next_bound[:,1] = np.clip(next_bound[:,1], -self.max_speed, self.max_speed)

        # Keep the raw theta interval. PendulumVerifier.split_merge_bounds
        # wraps it into valid state intervals before the next reachability step.
        return next_bound
    
class MountainCar(dynamic.MountainCar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, bound: np.ndarray):
        # Cosine term bounds: cos(3*pos)
        pos_bound = bound[:,0]
        vel_bound = bound[:,1]
        action_bound = bound[:,2]

        cos_3p = np.sort(np.cos(3.0 * pos_bound))[::-1]

        # Velocity update: v' = v + action*POWER - 0.0025*cos(3*pos)
        vel_bound = vel_bound + action_bound * self.power - 0.0025 * cos_3p

        # Velocity clipping: both bounds must respect speed limits
        vel_bound = np.clip(vel_bound, self.min_speed, self.max_speed)

        # Ensure velocity bounds are valid (min <= max)
        vel_bound = np.sort(vel_bound)

        # Position update: pos' = pos + v'
        pos_bound = pos_bound + vel_bound

        # Ensure position bounds are valid (min <= max)
        # if pos_next_min > pos_next_max:
        #     pos_next_min, pos_next_max = pos_next_max, pos_next_min

        return np.array([pos_bound, vel_bound]).T
    
class CartPole(dynamic.CartPole):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vel_bound = np.array([0.0,0.0])
        self.avel_bound = np.array([0.0,0.0])

    def cartpole(self, x, a):
        dxdt = [None] * 4
        a = a[0] * self.force_mag
        costheta = cos(x[2])
        sintheta = sin(x[2])
        temp = (a + self.polemass_length * x[3] * x[3] * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.mass_pole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        dxdt[0] = x[1]
        dxdt[1] = xacc
        dxdt[2] = x[3]
        dxdt[3] = thetaacc


        return Matrix(dxdt)

    def step(self, bound: np.ndarray) -> np.ndarray:

        action_bound = bound[:,-1]
        state_bound = bound[:,0:4]

        options = ASB2008CDC.Options()
        options.t_end = self.dt
        options.step = self.dt
        options.tensor_order = 3
        options.taylor_terms = 4
        u=Interval(action_bound[0],action_bound[1])
        options.u= cvt2(u, Geometry.TYPE.ZONOTOPE)
        options.u_trans = np.zeros(1)

        Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
        Zonotope.ORDER = 50

        z = Interval(
            state_bound[0],
            state_bound[1]
        )

        resolution = 0.15  # controls how finely the interval is decomposed
        xs = boundary(z, resolution, Geometry.TYPE.ZONOTOPE)

        lower=np.full(4, np.finfo(np.float32).max, dtype=np.float32)
        upper=np.full(4, np.finfo(np.float32).min, dtype=np.float32)
        for x in xs:
            ri_set, rp_set = ASB2008CDC.reach(
                lambda x, a: self.cartpole(x, a), 
                [4, 1], 
                options, 
                x
            )

            for i in ri_set:
                interval_i = cvt2(i, Geometry.TYPE.INTERVAL)
                # print(interval_i.inf)
                # print(interval_i.sup)
                
                for j in range(4):
                    lower[j]=min(lower[j], interval_i.inf[j])
                    upper[j]=max(upper[j], interval_i.sup[j])

        next_state_bound = np.array([lower,upper])

        return next_state_bound
    
