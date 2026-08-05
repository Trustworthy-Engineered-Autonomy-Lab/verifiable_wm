import csv
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
