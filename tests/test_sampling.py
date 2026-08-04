import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from make_decoder_dataset import save_real_trajectories


class SamplingOutputTests(unittest.TestCase):
    def test_saves_real_trajectory_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "output_dir": tmp,
                "rollout_steps": 1,
                "starv_config": "config/starv.json",
                "controller": {"weights": "weights/controller.pth"},
            }
            trajectory_splits = {
                "test": {
                    "traj": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                    "actions": torch.tensor([[[0.5]]]),
                }
            }

            save_real_trajectories(config, trajectory_splits)

            output_path = Path(tmp) / "real_trajectories.npz"
            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(data["rollout_steps"].item(), 1)
                self.assertEqual(data["starv_config"].item(), "config/starv.json")
                self.assertEqual(
                    data["controller_weights"].item(),
                    "weights/controller.pth",
                )


if __name__ == "__main__":
    unittest.main()
