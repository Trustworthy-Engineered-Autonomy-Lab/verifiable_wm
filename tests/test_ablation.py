import copy
import unittest
from pathlib import Path

from ablation import (
    DEFAULT_ALPHAS,
    DEFAULT_LAMBDAS,
    Experiment,
    build_experiments,
    build_train_config,
    format_value,
)


class AblationGridTests(unittest.TestCase):
    def test_complete_grid_has_49_unique_experiments(self):
        experiments = build_experiments("pendulum")

        self.assertEqual(len(experiments), 49)
        self.assertEqual(len(set(experiments)), 49)
        self.assertEqual({e.alpha for e in experiments}, set(DEFAULT_ALPHAS))
        self.assertEqual({e.lambda_ctrl for e in experiments}, set(DEFAULT_LAMBDAS))

    def test_paths_use_compact_decimal_names(self):
        experiment = Experiment("cartpole", 0.5, 0.001, 2025)

        self.assertEqual(format_value(8.0), "8")
        self.assertEqual(format_value(0.001), "0.001")
        self.assertEqual(
            experiment.output_dir,
            Path("dwm_weight/now_weight/cartpole/alpha_lambda_grid")
            / "alpha_0.5"
            / "lambda_0.001"
            / "seed_2025",
        )

    def test_build_train_config_does_not_mutate_base_config(self):
        base = {
            "weight_mode": "saliency",
            "weight": {"alpha": 8.0},
            "lambda_ctrl": 0.1,
            "training": {"seed": 7},
            "output_dir": "old",
        }
        original = copy.deepcopy(base)
        experiment = Experiment("pendulum", 2.0, 0.05, 2025)

        actual = build_train_config(experiment, base_config=base)

        self.assertEqual(base, original)
        self.assertEqual(actual["weight"]["alpha"], 2.0)
        self.assertEqual(actual["lambda_ctrl"], 0.05)
        self.assertEqual(actual["training"]["seed"], 2025)
        self.assertEqual(actual["output_dir"], experiment.output_dir.as_posix())


if __name__ == "__main__":
    unittest.main()
