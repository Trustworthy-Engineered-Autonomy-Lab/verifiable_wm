import unittest

import numpy as np
import torch

from saliency_map.methods import occlusion
from saliency_map.scripts.diagnostics.compare_occlusion_baselines import rank_mask
from saliency_map.scripts.precompute_saliency_maps import build_background_median


class PixelSumController(torch.nn.Module):
    def forward(self, images):
        return images.sum(dim=(1, 2, 3), keepdim=True)


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.controller = PixelSumController()
        self.images = torch.ones(2, 1, 4, 4)

    def test_background_tensor_replaces_each_patch_at_its_own_location(self):
        background = torch.zeros(1, 1, 4, 4)

        heat, _ = occlusion(
            self.controller,
            self.images,
            patch=2,
            stride=2,
            fill=background,
        )

        self.assertTrue(torch.allclose(heat, torch.full_like(heat, 4.0)))

    def test_background_tensor_requires_matching_image_geometry(self):
        background = torch.zeros(1, 1, 3, 4)

        with self.assertRaisesRegex(ValueError, "background"):
            occlusion(
                self.controller,
                self.images,
                patch=2,
                stride=2,
                fill=background,
            )

    def test_background_median_uses_train_images_only(self):
        train_images = np.array(
            [
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=np.float32,
        )
        data = {
            "train_images": train_images,
            "test_images": np.full_like(train_images, 99.0),
        }

        background = build_background_median(data)

        np.testing.assert_array_equal(
            background,
            np.median(train_images, axis=0, keepdims=True).astype(np.float32),
        )

    def test_rank_mask_selects_the_requested_number_of_top_bottom_and_random_pixels(self):
        heatmaps = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])

        top = rank_mask(heatmaps, kind="top", fraction=0.5, seed=7)
        bottom = rank_mask(heatmaps, kind="bottom", fraction=0.5, seed=7)
        random_a = rank_mask(heatmaps, kind="random", fraction=0.5, seed=7)
        random_b = rank_mask(heatmaps, kind="random", fraction=0.5, seed=7)

        self.assertEqual(int(top.sum()), 2)
        self.assertEqual(int(bottom.sum()), 2)
        self.assertTrue(torch.equal(top[0, 0], torch.tensor([[False, False], [True, True]])))
        self.assertTrue(torch.equal(bottom[0, 0], torch.tensor([[True, True], [False, False]])))
        self.assertTrue(torch.equal(random_a, random_b))


if __name__ == "__main__":
    unittest.main()
