import unittest

import numpy as np

from compare import validate_states_in_grid, validate_trajectory_initial_states
from tube_geometry import GridInfo


class CompareTrajectoryStateTests(unittest.TestCase):
    def test_accepts_one_full_state_set_shared_by_real_and_dwm(self):
        states = np.array([[0.1, 0.0, 0.08, 0.0]], dtype=float)
        real = np.zeros((1, 3, 4), dtype=float)
        dwm = np.zeros((1, 3, 4), dtype=float)
        real[:, 0, :] = states
        dwm[:, 0, :] = states

        actual = validate_trajectory_initial_states(states, real, dwm)

        np.testing.assert_array_equal(actual, states)

    def test_rejects_decoder_only_states_for_cell_lookup(self):
        states = np.array([[0.1, 0.08]], dtype=float)
        real = np.zeros((1, 3, 4), dtype=float)
        dwm = np.zeros((1, 3, 4), dtype=float)

        with self.assertRaisesRegex(ValueError, "full trajectory state"):
            validate_trajectory_initial_states(states, real, dwm)

    def test_rejects_trajectories_with_different_initial_states(self):
        states = np.array([[0.1, 0.0, 0.08, 0.0]], dtype=float)
        real = np.zeros((1, 3, 4), dtype=float)
        dwm = np.zeros((1, 3, 4), dtype=float)
        real[:, 0, :] = states
        dwm[:, 0, :] = states
        dwm[0, 0, 0] = 0.2

        with self.assertRaisesRegex(ValueError, "DWM trajectory t=0"):
            validate_trajectory_initial_states(states, real, dwm)

    def test_rejects_initial_states_outside_starv_grid(self):
        grid = GridInfo(
            names=["pos", "vel"],
            starts=np.array([0.0, 0.0]),
            stops=np.array([0.6, 0.0]),
            nums=np.array([60, 1]),
            steps=np.array([0.01, 0.0]),
        )

        with self.assertRaisesRegex(ValueError, "outside the StarV grid"):
            validate_states_in_grid(np.array([[0.7, 0.0]]), grid)


if __name__ == "__main__":
    unittest.main()
