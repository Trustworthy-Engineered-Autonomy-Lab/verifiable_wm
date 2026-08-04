"""MountainCar boundary semantics shared by rollout and StarV dynamics."""

import unittest

import numpy as np
import torch

import dynamic
import starv_verification.dynamic as starv_dynamic


class _DummyEnv:
    def close(self):
        pass


def make_point_mountain_car():
    model = dynamic.MountainCar.__new__(dynamic.MountainCar)
    model.min_pos, model.max_pos = -1.2, 0.6
    model.min_speed, model.max_speed = -0.07, 0.07
    model.min_action, model.max_action = -1.0, 1.0
    model.power = 0.0015
    model.env = _DummyEnv()
    return model


def make_starv_mountain_car():
    model = starv_dynamic.MountainCar.__new__(starv_dynamic.MountainCar)
    model.min_pos, model.max_pos = -1.2, 0.6
    model.min_speed, model.max_speed = -0.07, 0.07
    model.min_action, model.max_action = -1.0, 1.0
    model.power = 0.0015
    model.env = _DummyEnv()
    return model


class MountainCarPointTerminalTests(unittest.TestCase):
    def test_step_crossing_goal_is_absorbed_at_upper_boundary(self):
        model = make_point_mountain_car()

        stepped = model.step(
            torch.tensor([[0.59, 0.08]], dtype=torch.float64),
            torch.tensor([[0.0]], dtype=torch.float64),
        )

        torch.testing.assert_close(
            stepped,
            torch.tensor([[0.6, 0.0]], dtype=torch.float64),
        )

    def test_step_from_goal_remains_absorbed(self):
        model = make_point_mountain_car()

        stepped = model.step(
            torch.tensor([[0.6, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0]], dtype=torch.float64),
        )

        torch.testing.assert_close(
            stepped,
            torch.tensor([[0.6, 0.0]], dtype=torch.float64),
        )

    def test_step_clamps_left_wall_and_clears_negative_velocity(self):
        model = make_point_mountain_car()

        stepped = model.step(
            torch.tensor([[-1.19, -0.07]], dtype=torch.float64),
            torch.tensor([[-1.0]], dtype=torch.float64),
        )

        torch.testing.assert_close(
            stepped,
            torch.tensor([[-1.2, 0.0]], dtype=torch.float64),
        )


class MountainCarIntervalTerminalTests(unittest.TestCase):
    def test_interval_fully_crossing_goal_is_absorbed(self):
        model = make_starv_mountain_car()
        bound = np.array(
            [
                [0.59, 0.07, 0.0],
                [0.595, 0.07, 0.0],
            ],
            dtype=np.float64,
        )

        stepped = model.step(bound)

        np.testing.assert_allclose(
            stepped,
            np.array([[0.6, 0.0], [0.6, 0.0]]),
            atol=1e-12,
        )

    def test_partially_crossing_interval_contains_point_steps(self):
        point_model = make_point_mountain_car()
        interval_model = make_starv_mountain_car()
        bound = np.array(
            [
                [0.54, 0.00, -1.0],
                [0.59, 0.07, 1.0],
            ],
            dtype=np.float64,
        )

        stepped_bound = interval_model.step(bound)

        positions = np.linspace(bound[0, 0], bound[1, 0], 9)
        velocities = np.linspace(bound[0, 1], bound[1, 1], 9)
        actions = np.linspace(bound[0, 2], bound[1, 2], 5)
        samples = np.array(
            [
                (position, velocity, action)
                for position in positions
                for velocity in velocities
                for action in actions
            ]
        )
        point_steps = point_model.step(
            torch.tensor(samples[:, :2], dtype=torch.float64),
            torch.tensor(samples[:, 2:], dtype=torch.float64),
        ).numpy()

        self.assertLessEqual(float(stepped_bound[1, 0]), 0.6)
        self.assertLessEqual(float(stepped_bound[0, 1]), 0.0)
        self.assertGreaterEqual(float(stepped_bound[1, 1]), 0.0)
        for dimension in range(2):
            self.assertLessEqual(
                float(stepped_bound[0, dimension]),
                float(point_steps[:, dimension].min()) + 1e-12,
            )
            self.assertGreaterEqual(
                float(stepped_bound[1, dimension]),
                float(point_steps[:, dimension].max()) - 1e-12,
            )


if __name__ == "__main__":
    unittest.main()
