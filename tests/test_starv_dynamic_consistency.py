"""Consistency between the rollout dynamics (dynamic.py) and the StarV
interval dynamics (starv_verification/dynamic.py) for Pendulum.

The StarV class re-derives the same update equations in interval form, so
these tests pin the two implementations together: a degenerate interval
must reproduce the point dynamics, and a real interval must contain every
rollout step sampled from inside it.

Instances are built with __new__ + manual attributes to avoid constructing
gym render environments inside unit tests. lp_solver is 'linprog' so the
tests do not depend on a gurobi license.
"""

import unittest

import numpy as np
import torch

import dynamic
import starv_verification.dynamic as starv_dynamic


class _DummyEnv:
    def close(self):
        pass


def make_root_pendulum():
    p = dynamic.Pendulum.__new__(dynamic.Pendulum)
    p.max_speed, p.max_torque = 8.0, 2.0
    p.dt, p.g, p.m, p.l = 0.05, 10.0, 1.0, 1.0
    p.env = _DummyEnv()
    return p


def make_starv_pendulum(**overrides):
    p = starv_dynamic.Pendulum.__new__(starv_dynamic.Pendulum)
    p.max_speed = overrides.get("max_speed", 8.0)
    p.max_torque = overrides.get("max_torque", 2.0)
    p.dt, p.g, p.m, p.l = 0.05, 10.0, 1.0, 1.0
    p.lp_solver = "linprog"
    p.env = _DummyEnv()
    return p


class PendulumIntervalTests(unittest.TestCase):
    def test_degenerate_interval_matches_point_dynamics(self):
        root = make_root_pendulum()
        starv = make_starv_pendulum()

        for theta, omega, action in [
            (1.05, 4.55, 0.5),
            (-0.5, -2.0, -0.8),
            (0.2, 1.0, 1.0),
        ]:
            with self.subTest(theta=theta, omega=omega, action=action):
                point_next = root.step(
                    torch.tensor([[theta, omega]], dtype=torch.float64),
                    torch.tensor([[action]], dtype=torch.float64),
                ).numpy()[0]

                bound = np.array([
                    [theta, omega, action],
                    [theta, omega, action],
                ])
                interval_next = starv.step(bound)

                np.testing.assert_allclose(interval_next[0], point_next, atol=1e-5)
                np.testing.assert_allclose(interval_next[1], point_next, atol=1e-5)

    def test_interval_contains_sampled_rollout_steps(self):
        # Smoke-grid-sized cell with the full controller output range.
        # The rollout applies torque = clip(a, -1, 1) * max_torque(=2); a
        # tube modelling only half that torque cannot contain these steps.
        root = make_root_pendulum()
        starv = make_starv_pendulum()

        theta_lo, theta_hi = 1.03, 1.07
        omega_lo, omega_hi = 4.53, 4.57
        act_lo, act_hi = -1.0, 1.0
        bound = np.array([
            [theta_lo, omega_lo, act_lo],
            [theta_hi, omega_hi, act_hi],
        ])
        next_bound = starv.step(bound)

        rng = np.random.default_rng(2025)
        thetas = rng.uniform(theta_lo, theta_hi, size=300)
        omegas = rng.uniform(omega_lo, omega_hi, size=300)
        actions = rng.uniform(act_lo, act_hi, size=300)
        # Include the extreme-torque corners explicitly.
        thetas[0], omegas[0], actions[0] = theta_hi, omega_hi, act_hi
        thetas[1], omegas[1], actions[1] = theta_lo, omega_lo, act_lo

        states = torch.tensor(np.stack([thetas, omegas], axis=1))
        acts = torch.tensor(actions[:, None])
        stepped = root.step(states, acts).numpy()

        eps = 1e-4
        for dim in range(2):
            self.assertLessEqual(
                float(next_bound[0, dim]) - eps,
                float(stepped[:, dim].min()),
                f"dim {dim}: lower bound not conservative",
            )
            self.assertGreaterEqual(
                float(next_bound[1, dim]) + eps,
                float(stepped[:, dim].max()),
                f"dim {dim}: upper bound not conservative",
            )

    def test_torque_scaling_uses_max_torque_parameter(self):
        # Changing max_torque on the instance must change the omega update;
        # hard-coded coefficients would keep both results identical.
        weaker = make_starv_pendulum(max_torque=1.0)
        default = make_starv_pendulum(max_torque=2.0)

        bound = np.array([
            [1.05, 4.55, 1.0],
            [1.05, 4.55, 1.0],
        ])
        omega_weaker = float(weaker.step(bound.copy())[1, 1])
        omega_default = float(default.step(bound.copy())[1, 1])

        self.assertNotAlmostEqual(omega_weaker, omega_default, places=3)


if __name__ == "__main__":
    unittest.main()
