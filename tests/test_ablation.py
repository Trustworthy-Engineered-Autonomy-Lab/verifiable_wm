import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ablation import (
    DEFAULT_ALPHAS,
    DEFAULT_LAMBDAS,
    Experiment,
    build_experiments,
    build_train_config,
    format_value,
    run_training_grid,
    validate_training_artifacts,
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


class TrainingResumeTests(unittest.TestCase):
    def _experiment(self, root):
        return Experiment("pendulum", 2.0, 0.05, 2025, Path(root))

    def _write_complete_artifacts(self, experiment):
        experiment.output_dir.mkdir(parents=True)
        experiment.best_checkpoint.touch()
        experiment.last_checkpoint.touch()
        config = build_train_config(
            experiment,
            base_config={
                "weight_mode": "saliency",
                "weight": {"alpha": 8.0},
                "lambda_ctrl": 0.1,
                "training": {"seed": 7},
                "output_dir": "old",
            },
        )
        experiment.metrics_path.write_text(
            json.dumps(
                {
                    "config": config,
                    "best_checkpoint": "decoder_best_total.pth",
                }
            ),
            encoding="utf-8",
        )

    def test_complete_matching_training_artifacts_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = self._experiment(tmp)
            self._write_complete_artifacts(experiment)

            self.assertEqual(
                validate_training_artifacts(experiment),
                (True, "complete"),
            )

    def test_partial_training_artifacts_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = self._experiment(tmp)
            experiment.output_dir.mkdir(parents=True)
            experiment.best_checkpoint.touch()

            valid, reason = validate_training_artifacts(experiment)

            self.assertFalse(valid)
            self.assertIn("missing", reason)

    def test_mismatched_training_config_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = self._experiment(tmp)
            self._write_complete_artifacts(experiment)
            metrics = json.loads(experiment.metrics_path.read_text(encoding="utf-8"))
            metrics["config"]["lambda_ctrl"] = 0.1
            experiment.metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            self.assertEqual(
                validate_training_artifacts(experiment),
                (False, "config mismatch"),
            )

    def test_grid_runner_skips_valid_and_reports_failure(self):
        experiments = [
            Experiment("pendulum", 1.0, 0.0, 2025),
            Experiment("pendulum", 2.0, 0.0, 2025),
        ]
        with mock.patch("ablation.validate_training_artifacts") as validate:
            validate.side_effect = [(True, "complete"), (False, "missing")]
            trainer = mock.Mock(side_effect=RuntimeError("boom"))
            frame = run_training_grid(
                "pendulum",
                experiments=experiments,
                trainer=trainer,
                skip_existing=True,
                continue_on_error=True,
            )

        self.assertEqual(frame["status"].tolist(), ["skipped", "failed"])
        self.assertEqual(trainer.call_count, 1)
        self.assertIn("boom", frame.loc[1, "error"])


if __name__ == "__main__":
    unittest.main()
