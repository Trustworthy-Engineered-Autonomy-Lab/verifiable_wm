import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import signed_tube_margin as stm
from tools import plot_brake_dwm_sampled_diagnostic as diagnostic


class BrakeDwmSampledDiagnosticTests(unittest.TestCase):
    def test_build_diagnostic_selects_worst_real_and_pairs_model_index(self):
        grid = stm.GridInfo(
            names=["dis", "vel"],
            starts=np.array([0.0, 0.0]),
            stops=np.array([2.0, 1.0]),
            nums=np.array([2, 1]),
            steps=np.array([1.0, 1.0]),
        )
        cells = [
            {
                "bounds": [
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            },
            {
                "bounds": [
                    [[1.0, 2.0], [0.0, 1.0]],
                    [[1.0, 1.4], [0.0, 1.0]],
                ]
            },
        ]
        real = np.array(
            [
                [[0.5, 0.5], [0.6, 0.5]],
                [[1.5, 0.5], [1.8, 0.5]],
            ]
        )
        model = np.array(
            [
                [[0.5, 0.5], [0.61, 0.5]],
                [[1.5, 0.5], [1.7, 0.5]],
            ]
        )

        result = diagnostic.build_diagnostic(real, model, grid, cells)

        self.assertEqual(result["trajectory_index"], 1)
        self.assertEqual(result["cell_index"], 1)
        self.assertEqual(result["first_violating_step"], 1)
        np.testing.assert_allclose(result["model_states"], model[1])
        np.testing.assert_allclose(result["real_margins"], [0.5, -0.4])
        self.assertAlmostEqual(result["max_abs_state_difference"], 0.1)

    def test_write_diagnostic_creates_nonempty_png_and_exact_value_json(self):
        payload = {
            "trajectory_index": 1,
            "cell_index": 1,
            "first_violating_step": 1,
            "worst_real_margin": -0.4,
            "real_states": np.array([[1.5, 0.5], [1.8, 0.5]]),
            "model_states": np.array([[1.5, 0.5], [1.7, 0.5]]),
            "tube_bounds": [
                [[1.0, 2.0], [0.0, 1.0]],
                [[1.0, 1.4], [0.0, 1.0]],
            ],
            "real_margins": np.array([0.5, -0.4]),
            "model_margins": np.array([0.5, -0.3]),
            "state_difference": np.array([[0.0, 0.0], [0.1, 0.0]]),
            "max_abs_state_difference": 0.1,
            "seed": 728,
        }
        with tempfile.TemporaryDirectory() as tmp:
            png_path, json_path = diagnostic.write_diagnostic(
                payload,
                safety_path=Path("tube_seed_728.json"),
                real_path=Path("real.npz"),
                model_path=Path("dwm.npz"),
                output_dir=Path(tmp),
            )

            self.assertTrue(png_path.is_file())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertTrue(json_path.is_file())
            saved = json.loads(json_path.read_text())
            self.assertEqual(saved["seed"], 728)
            self.assertEqual(saved["trajectory_index"], 1)
            self.assertEqual(len(saved["real_margins"]), 2)


if __name__ == "__main__":
    unittest.main()
