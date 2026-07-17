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
    def __init__(self, lp_solver='gurobi', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lp_solver = lp_solver

    def angle_normalize(self, x):
        """Normalize angle to [-π, π] range"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def step(self, bound: np.ndarray):
        # theta' = theta + 0.05 * omega + 0.0075 * u + 0.0375 * sin(theta)
        # omega' = omega + 0.15 * u + 0.75 * sin(theta), clipped to [-8, 8]
        # Compute sin(theta) using SinLayer
        theta_min, theta_max = bound[:,0]

        L_sin = SinLayer()
        lb_theta = np.array([theta_min], dtype=np.float32)
        ub_theta = np.array([theta_max], dtype=np.float32)
        S_theta = Star(lb_theta, ub_theta)

        IM_sin = L_sin.reach(
            S_theta, method='approx', lp_solver=self.lp_solver, RF=0.0
        )
        try:
            z_bound = np.array(IM_sin.getRanges(lp_solver=self.lp_solver))
        except Exception as e:
            z_bound = np.array(IM_sin.getRanges(lp_solver='estimate'))

        full_bound = np.concatenate([bound, z_bound], axis = 1)
        S_full = Star(full_bound[0], full_bound[1])

        # Apply dynamics
        M = np.array([[1.0, 0.05, 0.0075, 0.0375],
                      [0.0, 1.0, 0.15, 0.75]], dtype=np.float32)
        b_dyn = np.zeros(2, dtype=np.float32)
        S_next = S_full.affineMap(M, b_dyn)

        # Step 7: Get bounds BEFORE clipping
        next_bound = np.array(S_next.getRanges(self.lp_solver, RF=0.0))

        # Clip omega' to [-8, 8]
        next_bound[:,1] = np.clip(next_bound[:,1], -8.0, 8.0)

        # Normalize theta to [-π, π]
        next_bound[:,0] = self.angle_normalize(next_bound[:,0])

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

    @staticmethod
    def cartpole(x, a):
        dxdt = [None] * 4
        a = a[0] * 10.0
        costheta = cos(x[2])
        sintheta = sin(x[2])
        temp = (a + 0.05 * x[3] * x[3] * sintheta) / 1.1
        thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0/3.0 - 0.1 * costheta * costheta / 1.1))
        xacc = temp - 0.05 * thetaacc * costheta / 1.1
        
        dxdt[0] = x[1]
        dxdt[1] = xacc
        dxdt[2] = x[3]
        dxdt[3] = thetaacc


        return Matrix(dxdt)

    def step(self, bound: np.ndarray) -> np.ndarray:

        action_bound = bound[:,-1]
        state_bound = bound[:,0:4]

        options = ASB2008CDC.Options()
        options.t_end = 0.02
        options.step = 0.02
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
                CartPole.cartpole, [4, 1], options, x
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
    
