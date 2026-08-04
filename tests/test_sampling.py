import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from make_decoder_dataset import save_real_trajectories
from sampling import save_dwm_trajectories


class SamplingOutputTests(unittest.TestCase):
    def test_saves_variant_specific_trajectory_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = "weights/decoder_best_total.pth"
            config = {
                "output_dir": tmp,
                "rollout_steps": 1,
                "starv_config": "config/starv.json",
                "controller": {"weights": "weights/controller.pth"},
                "decoder": {
                    "variant": "saliency",
                    "weights": {"saliency": weights},
                },
            }
            trajectory_splits = {
                "test": {
                    "traj": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                    "actions": torch.tensor([[[0.5]]]),
                }
            }

            save_dwm_trajectories(config, trajectory_splits)

            output_path = Path(tmp) / "dwm_trajectories_saliency.npz"
            self.assertTrue(output_path.is_file())
            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(data["variant"].item(), "saliency")
                self.assertEqual(data["decoder_weights"].item(), weights)
                self.assertEqual(data["rollout_steps"].item(), 1)
                self.assertEqual(data["starv_config"].item(), "config/starv.json")
                self.assertEqual(
                    data["controller_weights"].item(),
                    "weights/controller.pth",
                )
                np.testing.assert_array_equal(
                    data["test_traj"],
                    trajectory_splits["test"]["traj"].numpy(),
                )
                np.testing.assert_array_equal(
                    data["test_actions"],
                    trajectory_splits["test"]["actions"].numpy(),
                )

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
