import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import signed_tube_margin as stm
from tools import plot_brake_dwm_sampled_diagnostic as diagnostic


class BrakeDwmSampledDiagnosticTests(unittest.TestCase):
    def test_random_figure_layout_separates_title_legend_and_colorbar(self):
        layout = diagnostic.random_figure_layout()

        self.assertLess(layout["subplot_top"], layout["legend_y"])
        self.assertLess(layout["legend_y"], layout["title_y"])
        self.assertGreater(
            layout["colorbar_rect"][0], layout["subplot_right"]
        )
        self.assertLessEqual(
            layout["colorbar_rect"][0] + layout["colorbar_rect"][2], 1.0
        )

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

    def test_build_random_diagnostics_uses_fixed_seed_and_model_indices(self):
        grid = stm.GridInfo(
            names=["dis", "vel"],
            starts=np.array([0.0, 0.0]),
            stops=np.array([1.0, 1.0]),
            nums=np.array([1, 1]),
            steps=np.array([1.0, 1.0]),
        )
        cells = [
            {
                "bounds": [
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ]
            }
        ]
        real = np.array(
            [[[0.1 + 0.01 * index, 0.5], [0.2, 0.5]]
             for index in range(8)]
        )
        model = real + np.array([0.001, 0.0])
        expected = np.random.default_rng(728).choice(
            np.arange(8), 6, replace=False
        ).tolist()

        payloads = diagnostic.build_random_diagnostics(
            real, model, grid, cells, count=6, seed=728
        )

        self.assertEqual(
            [payload["trajectory_index"] for payload in payloads],
            expected,
        )
        for payload in payloads:
            index = payload["trajectory_index"]
            np.testing.assert_allclose(
                payload["model_states"], model[index]
            )

    def test_write_random_diagnostics_creates_six_trajectory_artifacts(self):
        payloads = []
        for index in range(6):
            payloads.append(
                {
                    "trajectory_index": index,
                    "cell_index": index,
                    "first_violating_step": 1,
                    "worst_real_margin": -0.1,
                    "real_states": np.array(
                        [[0.2, 0.5], [0.3, 0.49]]
                    ),
                    "model_states": np.array(
                        [[0.2, 0.5], [0.31, 0.49]]
                    ),
                    "tube_bounds": [
                        [[0.0, 0.4], [0.4, 0.6]],
                        [[0.1, 0.25], [0.4, 0.5]],
                    ],
                    "real_margins": np.array([0.1, -0.05]),
                    "model_margins": np.array([0.1, -0.06]),
                    "state_difference": np.array(
                        [[0.0, 0.0], [0.01, 0.0]]
                    ),
                    "max_abs_state_difference": 0.01,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            png_path, json_path = diagnostic.write_random_diagnostics(
                payloads,
                seed=728,
                safety_path=Path("tube_seed_728.json"),
                real_path=Path("real.npz"),
                model_path=Path("dwm.npz"),
                output_dir=Path(tmp),
            )

            self.assertTrue(png_path.is_file())
            self.assertGreater(png_path.stat().st_size, 0)
            saved = json.loads(json_path.read_text())
            self.assertEqual(saved["seed"], 728)
            self.assertEqual(len(saved["trajectories"]), 6)
            self.assertEqual(
                saved["selected_trajectory_indices"], list(range(6))
            )


if __name__ == "__main__":
    unittest.main()
