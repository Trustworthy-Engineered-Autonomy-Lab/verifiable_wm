import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils import (
    load_state_splits,
    sample_starv_state_splits,
    starv_grid_to_state_space,
)


class StarvStateTests(unittest.TestCase):
    def setUp(self):
        self.starv_config = {
            "grid": {
                "dims": [
                    {"name": "pos", "start": 0.0, "stop": 0.6, "num": 60},
                    {"name": "vel", "start": 0.0, "stop": 0.0, "num": 1},
                    {"name": "angle", "start": 0.06, "stop": 0.12, "num": 60},
                    {"name": "avel", "start": 0.0, "stop": 0.0, "num": 1},
                ]
            },
            "starv_states": {
                "num_train": 5,
                "num_val": 3,
                "num_test": 4,
                "seed_train": 11,
                "seed_val": 12,
                "seed_test": 13,
                "output_file": "starv_states.npz",
            },
        }

    def test_starv_grid_is_the_trajectory_range_source(self):
        state_space = starv_grid_to_state_space(self.starv_config)
        self.assertEqual(
            state_space,
            [
                {"name": "pos", "low": 0.0, "high": 0.6},
                {"name": "vel", "low": 0.0, "high": 0.0},
                {"name": "angle", "low": 0.06, "high": 0.12},
                {"name": "avel", "low": 0.0, "high": 0.0},
            ],
        )

    def test_sampled_states_are_full_dimensional_and_inside_grid(self):
        splits = sample_starv_state_splits(self.starv_config, torch.device("cpu"))

        self.assertEqual(tuple(splits["train_states"].shape), (5, 4))
        self.assertEqual(tuple(splits["val_states"].shape), (3, 4))
        self.assertEqual(tuple(splits["test_states"].shape), (4, 4))

        test_states = splits["test_states"].numpy()
        self.assertTrue(np.all((0.0 <= test_states[:, 0]) & (test_states[:, 0] <= 0.6)))
        self.assertTrue(np.all(test_states[:, 1] == 0.0))
        self.assertTrue(np.all((0.06 <= test_states[:, 2]) & (test_states[:, 2] <= 0.12)))
        self.assertTrue(np.all(test_states[:, 3] == 0.0))

    def test_saved_splits_load_without_resampling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "starv_states.npz"
            expected = {
                "train_states": np.array([[0.1, 0.0]], dtype=np.float32),
                "val_states": np.array([[0.2, 0.0]], dtype=np.float32),
                "test_states": np.array([[0.3, 0.0]], dtype=np.float32),
            }
            np.savez_compressed(path, **expected)

            actual = load_state_splits(path, torch.device("cpu"))

            for key, value in expected.items():
                np.testing.assert_array_equal(actual[key].numpy(), value)


if __name__ == "__main__":
    unittest.main()
