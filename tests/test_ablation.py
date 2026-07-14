import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ablation import (
    DEFAULT_ALPHAS,
    DEFAULT_LAMBDAS,
    Experiment,
    build_experiments,
    build_sampling_config,
    build_combined_metrics,
    build_train_config,
    compute_l2_metrics,
    format_value,
    run_mainline_rollouts,
    run_rollout_grid,
    run_training_grid,
    pivot_metric,
    validate_rollout_artifact,
    validate_training_artifacts,
    write_summary_tables,
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


class RolloutResumeTests(unittest.TestCase):
    def _write_rollout_fixture(self, tmp, decoder_weights):
        experiment = Experiment("pendulum", 8.0, 0.1, 2025, Path(tmp))
        experiment.output_dir.mkdir(parents=True)
        states_path = Path(tmp) / "starv_states.npz"
        states = np.array([[1.0, 4.5]], dtype=np.float32)
        np.savez_compressed(
            states_path,
            train_states=states,
            val_states=states,
            test_states=states,
        )
        np.savez_compressed(
            experiment.trajectory_path,
            train_traj=states[:, None],
            train_actions=np.zeros((1, 0, 1)),
            val_traj=states[:, None],
            val_actions=np.zeros((1, 0, 1)),
            test_traj=states[:, None],
            test_actions=np.zeros((1, 0, 1)),
            variant=np.array("saliency"),
            decoder_weights=np.array(str(decoder_weights)),
        )
        return experiment, states_path

    def test_sampling_config_uses_direct_grid_checkpoint_and_output_dir(self):
        experiment = Experiment("cartpole", 4.0, 0.1, 2025)
        base = {
            "decoder": {
                "name": "Decoder",
                "variant": "old",
                "weights": {"old": "x"},
            },
            "output_dir": "datasets/cartpole/data/dataset_v1",
            "starv_config": "config/starv_verification/cartpole.json",
        }

        actual = build_sampling_config(experiment, base_config=base)

        self.assertEqual(actual["decoder"]["variant"], "saliency")
        self.assertEqual(
            actual["decoder"]["weights"],
            experiment.best_checkpoint.as_posix(),
        )
        self.assertEqual(actual["output_dir"], experiment.output_dir.as_posix())

    def test_rollout_validation_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment, states_path = self._write_rollout_fixture(tmp, "wrong.pth")

            valid, reason = validate_rollout_artifact(
                experiment,
                states_path=states_path,
            )

            self.assertFalse(valid)
            self.assertIn("checkpoint", reason)

    def test_rollout_validation_rejects_changed_initial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment("pendulum", 8.0, 0.1, 2025, Path(tmp))
            experiment, states_path = self._write_rollout_fixture(
                tmp,
                experiment.best_checkpoint,
            )
            with np.load(experiment.trajectory_path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            arrays["val_traj"] = arrays["val_traj"].copy()
            arrays["val_traj"][0, 0, 0] += 1.0
            np.savez_compressed(experiment.trajectory_path, **arrays)

            valid, reason = validate_rollout_artifact(
                experiment,
                states_path=states_path,
            )

            self.assertFalse(valid)
            self.assertIn("val initial state", reason)

    def test_rollout_validation_accepts_matching_initial_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment("pendulum", 8.0, 0.1, 2025, Path(tmp))
            experiment, states_path = self._write_rollout_fixture(
                tmp,
                experiment.best_checkpoint,
            )

            self.assertEqual(
                validate_rollout_artifact(experiment, states_path=states_path),
                (True, "complete"),
            )

    def test_rollout_grid_does_not_generate_for_incomplete_training(self):
        experiment = Experiment("cartpole", 1.0, 0.0, 2025)
        generator = mock.Mock()
        with mock.patch(
            "ablation.validate_training_artifacts",
            return_value=(False, "missing checkpoint"),
        ):
            frame = run_rollout_grid(
                "cartpole",
                experiments=[experiment],
                generator=generator,
            )

        generator.assert_not_called()
        self.assertEqual(frame.loc[0, "status"], "failed")
        self.assertIn("missing checkpoint", frame.loc[0, "error"])

    def test_mainline_rollout_selects_requested_variant(self):
        base = {
            "decoder": {
                "variant": "old",
                "weights": {
                    "intensity": "weights/intensity.pth",
                    "saliency": "weights/saliency.pth",
                },
            },
            "output_dir": "datasets/cartpole/data/dataset_v1",
        }
        generator = mock.Mock()
        with mock.patch("ablation.load_config", return_value=base), mock.patch(
            "ablation._starv_states_path_for_env",
            return_value=Path("states.npz"),
        ), mock.patch(
            "ablation._validate_trajectory_file",
            side_effect=[(False, "missing"), (True, "complete")],
        ):
            frame = run_mainline_rollouts(
                "cartpole",
                variants=("saliency",),
                generator=generator,
            )

        called_config = generator.call_args.args[0]
        self.assertEqual(called_config["decoder"]["variant"], "saliency")
        self.assertEqual(base["decoder"]["variant"], "old")
        self.assertEqual(frame.loc[0, "status"], "generated")


class L2MetricTests(unittest.TestCase):
    def test_cartpole_full_state_l2(self):
        real = np.zeros((1, 3, 4), dtype=float)
        dwm = real.copy()
        dwm[0, 1, :] = np.array([1.0, 2.0, 2.0, 0.0])

        actual = compute_l2_metrics(real, dwm)

        self.assertAlmostEqual(actual["mean_step_l2"], 1.0)
        self.assertAlmostEqual(actual["final_l2"], 0.0)
        self.assertAlmostEqual(actual["max_l2_mean"], 3.0)
        self.assertAlmostEqual(actual["max_l2_p95"], 3.0)

    def test_pendulum_theta_uses_short_circular_difference(self):
        real = np.array([[[0.0, 0.0], [np.pi - 0.001, 0.0]]])
        dwm = np.array([[[0.0, 0.0], [-np.pi + 0.001, 0.0]]])

        actual = compute_l2_metrics(real, dwm, circular_dims=(0,))

        self.assertAlmostEqual(actual["max_l2_mean"], 0.002, places=6)

    def test_l2_rejects_different_initial_states(self):
        real = np.zeros((1, 2, 2))
        dwm = real.copy()
        dwm[0, 0, 0] = 1.0

        with self.assertRaisesRegex(ValueError, "initial states"):
            compute_l2_metrics(real, dwm)

    def test_combined_table_joins_training_and_rollout_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = Experiment(
                "pendulum",
                2.0,
                0.05,
                2025,
                root / "weights",
            )
            experiment.output_dir.mkdir(parents=True)
            metrics = {
                "config": {
                    "weight_mode": "saliency",
                    "weight": {"alpha": 2.0},
                    "lambda_ctrl": 0.05,
                    "training": {"seed": 2025},
                    "output_dir": experiment.output_dir.as_posix(),
                },
                "best_epoch": 1,
                "best_checkpoint": "decoder_best_total.pth",
                "history": [
                    {"epoch": 0, "val_ctrl_mse": 0.3, "val_pixel_mse": 0.4},
                    {"epoch": 1, "val_ctrl_mse": 0.1, "val_pixel_mse": 0.2},
                ],
                "test": {"ctrl_mse": 0.11, "pixel_mse": 0.21},
            }
            experiment.metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            experiment.best_checkpoint.touch()
            experiment.last_checkpoint.touch()

            initial = np.array([[1.0, 4.5]], dtype=np.float32)
            real_val = np.stack(
                [initial, initial + np.array([[0.1, 0.2]])],
                axis=1,
            )
            real_test = np.stack(
                [initial, initial + np.array([[0.2, 0.3]])],
                axis=1,
            )
            real_path = root / "real_trajectories.npz"
            np.savez_compressed(
                real_path,
                val_traj=real_val,
                test_traj=real_test,
            )
            np.savez_compressed(
                experiment.trajectory_path,
                val_traj=real_val.copy(),
                test_traj=real_test.copy(),
                variant=np.array("saliency"),
                decoder_weights=np.array(experiment.best_checkpoint.as_posix()),
            )

            combined = build_combined_metrics(
                "pendulum",
                experiments=[experiment],
                real_path=real_path,
            )

            self.assertEqual(len(combined), 2)
            self.assertEqual(set(combined["split"]), {"val", "test"})
            self.assertEqual(
                set(combined.columns),
                {
                    "env",
                    "alpha",
                    "lambda_ctrl",
                    "seed",
                    "split",
                    "best_epoch",
                    "ctrl_mse",
                    "pixel_mse",
                    "mean_step_l2",
                    "final_l2",
                    "max_l2_mean",
                    "max_l2_p95",
                },
            )

            with mock.patch("ablation.GRID_ROOT", root / "summaries"):
                result = write_summary_tables(
                    "pendulum",
                    experiments=[experiment],
                    real_path=real_path,
                )
            round_trip = pd.read_csv(result["paths"]["combined"])
            self.assertEqual(round_trip.columns.tolist(), combined.columns.tolist())
            self.assertEqual(len(round_trip), len(combined))

    def test_pivot_uses_alpha_rows_and_lambda_columns(self):
        frame = pd.DataFrame(
            [
                {"split": "val", "alpha": 2.0, "lambda_ctrl": 0.1, "score": 3.0},
                {"split": "val", "alpha": 1.0, "lambda_ctrl": 0.1, "score": 2.0},
            ]
        )

        pivot = pivot_metric(frame, "score")

        self.assertEqual(pivot.index.tolist(), [1.0, 2.0])
        self.assertEqual(pivot.columns.tolist(), [0.1])


if __name__ == "__main__":
    unittest.main()
