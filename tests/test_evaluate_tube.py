import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import evaluate_tube as evaluation


class EvaluateTubeTests(unittest.TestCase):
    def make_grid(self):
        return evaluation.GridInfo(
            names=["x", "y"],
            starts=np.array([0.0, 0.0]),
            stops=np.array([1.0, 1.0]),
            nums=np.array([1, 1]),
            steps=np.array([1.0, 1.0]),
        )

    def test_calculate_metrics_returns_only_scalar_coverage_and_area(self):
        cells = [{
            "bounds": [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 2.0], [0.0, 3.0]],
            ],
            "result": True,
        }]
        trajectories = np.array([
            [[0.5, 0.5], [1.0, 1.0]],
            [[0.5, 0.5], [2.5, 1.0]],
        ])

        metrics, rows = evaluation.calculate_metrics(
            trajectories, self.make_grid(), cells, dims=(0, 1)
        )

        self.assertEqual(metrics, {"coverage": 0.5, "area": 6.0})
        self.assertEqual([row["fully_inside"] for row in rows], [True, False])

    def test_area_excludes_the_initial_cell(self):
        cells = [{
            "bounds": [
                [[0.0, 100.0], [0.0, 100.0]],
                [[0.0, 2.0], [0.0, 3.0]],
                [[0.0, 4.0], [0.0, 2.0]],
            ]
        }]

        self.assertEqual(evaluation.average_tube_area(cells, dims=(0, 1)), 7.0)

    def test_invalid_trajectory_raises_instead_of_changing_denominator(self):
        cells = [{"bounds": [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]}]
        trajectories = np.array([[[2.0, 2.0], [2.0, 2.0]]])

        with self.assertRaisesRegex(ValueError, "trajectory 0"):
            evaluation.calculate_metrics(
                trajectories, self.make_grid(), cells, dims=(0, 1)
            )

    def test_write_metrics_table_uses_display_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tube_metrics.csv"
            evaluation.write_metrics_table(
                path, {"coverage": 0.9425, "area": 0.0012935951128641556}
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows, [{"Coverage": "94.25%", "Area": "0.001294"}])

    def test_load_postprocess_config_resolves_evaluation_paths_from_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cartpole.json"
            config_path.write_text(json.dumps({
                "tube": "safety/raw.json",
                "real_trajectories": "safety/real.npz",
                "check_dims": [0, 2],
                "evaluation": {
                    "real_key": "test_traj",
                    "output_dir": "results/evaluation",
                },
                "inflation": {
                    "calibration_key": "val_traj",
                    "alpha": 0.05,
                    "output": "results/inflated.json",
                    "calibration_output": "results/calibration.json",
                },
            }), encoding="utf-8")

            config = evaluation.load_postprocess_config(
                config_path, "evaluation", project_root=root
            )

        self.assertEqual(config, {
            "tube_path": root / "safety/raw.json",
            "real_path": root / "safety/real.npz",
            "real_key": "test_traj",
            "dims": (0, 2),
            "outdir": root / "results/evaluation",
        })

    def test_load_postprocess_config_names_a_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "broken.json"
            config_path.write_text(json.dumps({
                "tube": "raw.json",
                "real_trajectories": "real.npz",
                "check_dims": [0, 1],
                "evaluation": {"real_key": "test_traj"},
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "evaluation.output_dir"):
                evaluation.load_postprocess_config(
                    config_path, "evaluation", project_root=Path(tmp)
                )

    def test_main_reads_evaluation_section_from_positional_config(self):
        loaded = {
            "tube_path": Path("raw.json"),
            "real_path": Path("real.npz"),
            "real_key": "test_traj",
            "dims": (0, 1),
            "outdir": Path("out"),
        }
        with mock.patch.object(
            evaluation, "load_postprocess_config", return_value=loaded
        ) as load_config, mock.patch.object(
            evaluation, "run_evaluation", return_value={"coverage": 0.9, "area": 1.2}
        ) as run:
            evaluation.main(["config.json", "--overwrite"])

        load_config.assert_called_once_with(Path("config.json"), "evaluation")
        run.assert_called_once_with(**loaded, overwrite=True)


if __name__ == "__main__":
    unittest.main()
