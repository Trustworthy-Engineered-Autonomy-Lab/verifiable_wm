import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sampling import save_dwm_trajectories


class SamplingOutputTests(unittest.TestCase):
    def test_saves_variant_specific_trajectory_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = "weights/decoder_best_total.pth"
            config = {
                "output_dir": tmp,
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
                np.testing.assert_array_equal(
                    data["test_traj"],
                    trajectory_splits["test"]["traj"].numpy(),
                )
                np.testing.assert_array_equal(
                    data["test_actions"],
                    trajectory_splits["test"]["actions"].numpy(),
                )


if __name__ == "__main__":
    unittest.main()
