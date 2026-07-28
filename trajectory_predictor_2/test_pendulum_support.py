#!/usr/bin/env python3
"""Tests for periodic Pendulum angles and cell sampling."""

from __future__ import annotations

import unittest

import numpy as np

from angle_utils import (
    periodic_interval_to_json,
    unwrap_angle_trajectories,
    wrap_angle_trajectories,
)
from sampling_utils import sample_cell


class PendulumAngleTests(unittest.TestCase):
    def test_unwrap_removes_branch_cut_jump(self) -> None:
        trajectories = np.asarray(
            [[
                [3.10, 0.0],
                [3.13, 0.0],
                [-3.12, 0.0],
                [-3.08, 0.0],
            ]],
            dtype=np.float32,
        )
        unwrapped = unwrap_angle_trajectories(trajectories)
        increments = np.diff(unwrapped[0, :, 0])
        self.assertTrue(np.all(np.abs(increments) < 0.1))
        self.assertGreater(unwrapped[0, 2, 0], np.pi)

    def test_wrap_restores_canonical_range(self) -> None:
        trajectories = np.asarray(
            [[[3.10, 1.0], [3.20, 2.0], [-3.30, 3.0]]],
            dtype=np.float32,
        )
        wrapped = wrap_angle_trajectories(trajectories)
        self.assertTrue(np.all(wrapped[..., 0] >= -np.pi))
        self.assertTrue(np.all(wrapped[..., 0] < np.pi))
        np.testing.assert_allclose(wrapped[..., 1], trajectories[..., 1])

    def test_crossing_interval_becomes_union(self) -> None:
        interval = periodic_interval_to_json(3.05, 3.20)
        self.assertEqual(len(interval), 4)
        self.assertAlmostEqual(interval[0], 3.05)
        self.assertAlmostEqual(interval[1], np.pi)
        self.assertAlmostEqual(interval[2], -np.pi)
        self.assertLess(interval[3], -3.0)

    def test_non_crossing_interval_stays_single(self) -> None:
        interval = periodic_interval_to_json(-1.0, 0.5)
        self.assertEqual(len(interval), 2)
        np.testing.assert_allclose(interval, [-1.0, 0.5])

    def test_zero_width_at_pi_is_not_full_circle(self) -> None:
        interval = periodic_interval_to_json(np.pi, np.pi)
        self.assertEqual(len(interval), 2)
        np.testing.assert_allclose(interval, [-np.pi, -np.pi])


class CellSamplingTests(unittest.TestCase):
    def test_two_active_dimensions_get_corners_and_center(self) -> None:
        bounds = np.asarray([[-1.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        points = sample_cell(bounds, samples_per_cell=5)
        expected = np.asarray(
            [
                [-1.0, 2.0],
                [-1.0, 4.0],
                [1.0, 2.0],
                [1.0, 4.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(points.shape, (5, 2))
        for point in expected:
            self.assertTrue(any(np.allclose(point, actual) for actual in points))

    def test_too_few_samples_is_rejected(self) -> None:
        bounds = np.asarray([[-1.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "requires at least 5"):
            sample_cell(bounds, samples_per_cell=3)


if __name__ == "__main__":
    unittest.main()
