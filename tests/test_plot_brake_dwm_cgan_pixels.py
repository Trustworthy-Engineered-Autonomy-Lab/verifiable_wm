import unittest

import numpy as np

from tools.plot_brake_dwm_cgan_pixels import absolute_difference


class PixelComparisonTests(unittest.TestCase):
    def test_absolute_difference_is_elementwise_and_preserves_shape(self):
        dwm = np.array([[0.0, 0.75]], dtype=np.float32)
        cgan = np.array([[1.0, 0.25]], dtype=np.float32)

        result = absolute_difference(dwm, cgan)

        np.testing.assert_allclose(result, [[1.0, 0.5]])
        self.assertEqual(result.shape, (1, 2))

    def test_absolute_difference_rejects_different_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            absolute_difference(np.zeros((2, 2)), np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
