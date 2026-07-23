import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from conformal import calibrate_delta, conformal_quantile, nonconformity_scores


class ConformalQuantileTests(unittest.TestCase):
    def test_uses_ceil_n_plus_1_times_1_minus_alpha_order_statistic(self):
        # n=5, alpha=0.4 -> k = ceil(6 * 0.6) = ceil(3.6) = 4th smallest (1-indexed).
        scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

        delta = conformal_quantile(scores, alpha=0.4)

        self.assertEqual(delta, 4.0)

    def test_returns_infinity_when_confidence_exceeds_calibration_set_size(self):
        # n=5, alpha=0.01 -> k = ceil(6 * 0.99) = 6 > n, no finite guarantee.
        scores = np.array([5.0, 1.0, 4.0, 2.0, 3.0])

        delta = conformal_quantile(scores, alpha=0.01)

        self.assertEqual(delta, math.inf)

    def test_rejects_empty_calibration_set(self):
        with self.assertRaises(ValueError):
            conformal_quantile(np.array([]), alpha=0.1)


class NonconformityScoreTests(unittest.TestCase):
    def test_computes_max_over_time_of_l1_distance_on_selected_dims(self):
        # traj shape (N=1, T=2, dim=3); only dims (0, 2) are checked.
        real = np.array([[[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]]])
        dwm = np.array([[[1.0, -999.0, 1.0], [3.0, -999.0, 4.0]]])

        scores = nonconformity_scores(real, dwm, dims=(0, 2))

        # t=0: |1-0|+|1-0|=2 ; t=1: |3-0|+|4-0|=7 ; max over time = 7.
        self.assertEqual(scores.tolist(), [7.0])

    def test_wraps_circular_dims_to_the_shorter_angular_distance(self):
        real = np.array([[[3.13], [3.13]]])
        dwm = np.array([[[-3.13], [-3.13]]])

        wrapped = nonconformity_scores(
            real, dwm, dims=(0,), circular_dims=(0,), period=2 * math.pi
        )
        unwrapped = nonconformity_scores(real, dwm, dims=(0,))

        self.assertAlmostEqual(wrapped[0], 2 * math.pi - 2 * 3.13, places=5)
        self.assertGreater(unwrapped[0], wrapped[0])


class CalibrateDeltaTests(unittest.TestCase):
    def test_calibrates_from_val_split_and_reports_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_path = root / "real_trajectories.npz"
            dwm_path = root / "dwm_trajectories_saliency.npz"

            # 5 val trajectories, T=1 step, dim=2; deviations on dim 0 are 1..5.
            val_real = np.zeros((5, 2, 2), dtype=np.float32)
            val_dwm = val_real.copy()
            val_dwm[:, 1, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

            np.savez_compressed(real_path, val_traj=val_real, test_traj=val_real)
            np.savez_compressed(dwm_path, val_traj=val_dwm, test_traj=val_dwm)

            result = calibrate_delta(
                real_path=real_path,
                dwm_path=dwm_path,
                dims=(0, 1),
                alpha=0.4,
                split="val",
            )

            self.assertEqual(result["n"], 5)
            self.assertEqual(result["alpha"], 0.4)
            self.assertEqual(result["delta"], 4.0)
            self.assertEqual(result["split"], "val")


if __name__ == "__main__":
    unittest.main()
