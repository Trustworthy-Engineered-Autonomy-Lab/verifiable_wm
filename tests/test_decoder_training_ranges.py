import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DecoderTrainingRangeTests(unittest.TestCase):
    def test_decoder_training_ranges_are_broader_than_starv_grids(self):
        expected_ranges = {
            "cartpole": [
                (-2.4, 2.4),
                (-1.0, 1.0),
                (-0.2095, 0.2095),
                (-1.0, 1.0),
            ],
            "pendulum": [
                (-math.pi, math.pi),
                (-8.0, 8.0),
            ],
            "mountain_car": [
                (-1.2, 0.6),
                (-0.08, 0.08),
            ],
        }

        for environment, expected_range in expected_ranges.items():
            with self.subTest(environment=environment):
                make_config = self._load_json(
                    f"config/make_decoder_dataset/{environment}.json"
                )
                sampling_config = self._load_json(
                    f"config/sampling/{environment}.json"
                )
                starv_config = self._load_json(sampling_config["starv_config"])

                actual_range = [
                    (dimension["low"], dimension["high"])
                    for dimension in make_config["state_space"]
                ]
                # Tolerance instead of exact equality: importing the StarV
                # stack (pybdr -> codac) changes the process FPU rounding
                # mode, which perturbs all later JSON float parsing by 1 ulp.
                self.assertEqual(len(actual_range), len(expected_range))
                for actual, expected in zip(actual_range, expected_range):
                    for actual_value, expected_value in zip(actual, expected):
                        self.assertTrue(
                            math.isclose(
                                actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
                            ),
                            f"{environment}: {actual_range} != {expected_range}",
                        )

                for training_dim, grid_dim in zip(
                    make_config["state_space"], starv_config["grid"]["dims"]
                ):
                    self.assertLessEqual(training_dim["low"], grid_dim["start"])
                    self.assertGreaterEqual(training_dim["high"], grid_dim["stop"])

    @staticmethod
    def _load_json(relative_path):
        with (ROOT / relative_path).open() as file:
            return json.load(file)


if __name__ == "__main__":
    unittest.main()
