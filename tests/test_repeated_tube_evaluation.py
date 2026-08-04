import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import repeated_tube_evaluation as rte


class RepeatedTubeEvaluationModuleTests(unittest.TestCase):
    def test_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("repeated_tube_evaluation"),
            "repeated_tube_evaluation.py must provide the repeated evaluation entry point",
        )

    def test_shared_inputs_use_the_cell_count_names(self):
        self.assertEqual(
            rte.ENVIRONMENTS["cartpole"]["real"].name,
            "3600cell_real_trajectories.npz",
        )
        self.assertEqual(
            rte.ENVIRONMENTS["cartpole"]["symbolic"]["dwm"].name,
            "3600cell_dwm_safety_result.json",
        )
        self.assertEqual(
            rte.ENVIRONMENTS["cartpole"]["symbolic"]["cgan"].name,
            "3600cell_g_mlp_safety_result.json",
        )
        self.assertEqual(
            rte.ENVIRONMENTS["brake_system"]["real"].name,
            "1600cell_real_trajectories.npz",
        )
        self.assertEqual(
            rte.ENVIRONMENTS["brake_system"]["symbolic"]["dwm"].name,
            "1600cell_dwm_safety_result.json",
        )
        self.assertEqual(
            rte.ENVIRONMENTS["brake_system"]["symbolic"]["cgan"].name,
            "1600cell_g_mlp_safety_result.json",
        )
        self.assertEqual(
            rte.ENVIRONMENTS["mountain_car"]["real"].name,
            "6400cell_real_trajectories.npz",
        )
        self.assertEqual(rte.ENVIRONMENTS["mountain_car"]["expected_cells"], 6400)
        self.assertEqual(
            rte.ENVIRONMENTS["pendulum"]["real"].name,
            "5000cell_real_trajectories.npz",
        )
        self.assertEqual(rte.ENVIRONMENTS["pendulum"]["expected_cells"], 5000)

    def test_text_artifact_reuses_identical_content_and_rejects_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            rte.write_text_artifact(path, "header\nvalue\n", overwrite=False)
            rte.write_text_artifact(path, "header\nvalue\n", overwrite=False)
            with self.assertRaisesRegex(FileExistsError, "differs"):
                rte.write_text_artifact(path, "different\n", overwrite=False)


class SeedAndSplitTests(unittest.TestCase):
    def test_repeat_seeds_are_deterministic_independent_and_unique(self):
        first = rte.derive_repeat_seeds(2025, num_repeats=5)
        second = rte.derive_repeat_seeds(2025, num_repeats=5)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual([row["repeat_index"] for row in first], list(range(5)))
        self.assertEqual(len({row["split_seed"] for row in first}), 5)
        self.assertEqual(len({row["tube_seed"] for row in first}), 5)
        self.assertTrue(all(row["split_seed"] != row["tube_seed"] for row in first))

    def test_pool_concatenates_exactly_400_val_and_400_test_trajectories(self):
        val = np.arange(400 * 3 * 2, dtype=float).reshape(400, 3, 2)
        test = val + 10000.0

        pool = rte.combine_real_pool(val, test)

        self.assertEqual(pool.shape, (800, 3, 2))
        np.testing.assert_array_equal(pool[:400], val)
        np.testing.assert_array_equal(pool[400:], test)

    def test_split_stores_a_disjoint_600_200_partition_and_actual_arrays(self):
        pool = np.arange(800 * 2, dtype=float).reshape(800, 1, 2)

        split = rte.split_real_pool(pool, split_seed=1234)

        self.assertEqual(split["val_traj"].shape, (600, 1, 2))
        self.assertEqual(split["test_traj"].shape, (200, 1, 2))
        self.assertEqual(len(np.unique(split["val_indices"])), 600)
        self.assertEqual(len(np.unique(split["test_indices"])), 200)
        self.assertEqual(
            set(split["val_indices"]) | set(split["test_indices"]),
            set(range(800)),
        )
        self.assertFalse(
            set(split["val_indices"]) & set(split["test_indices"])
        )
        np.testing.assert_array_equal(
            split["val_traj"], pool[split["val_indices"]]
        )
        np.testing.assert_array_equal(
            split["test_traj"], pool[split["test_indices"]]
        )

    def test_split_artifact_round_trips_trajectories_indices_and_metadata(self):
        pool = np.arange(800 * 2, dtype=float).reshape(800, 1, 2)
        split = rte.split_real_pool(pool, split_seed=4321)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repeat_00.npz"
            rte.write_split_artifact(
                path,
                split,
                repeat_index=0,
                pool_fingerprint="abc123",
            )

            saved = rte.load_split_artifact(
                path,
                expected_repeat=0,
                expected_seed=4321,
                expected_pool_fingerprint="abc123",
            )

        np.testing.assert_array_equal(saved["val_traj"], split["val_traj"])
        np.testing.assert_array_equal(saved["test_traj"], split["test_traj"])
        np.testing.assert_array_equal(saved["val_indices"], split["val_indices"])
        np.testing.assert_array_equal(saved["test_indices"], split["test_indices"])

    def test_prepare_environment_writes_pool_and_five_reusable_splits(self):
        val = np.arange(400 * 2 * 2, dtype=float).reshape(400, 2, 2)
        test = val + 10000.0
        seeds = rte.derive_repeat_seeds(2025)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "real.npz"
            np.savez_compressed(source, val_traj=val, test_traj=test)
            env_root = root / "experiment" / "cartpole"

            paths = rte.prepare_environment_splits(
                source,
                env_root,
                seeds,
                overwrite=False,
            )
            repeated = rte.prepare_environment_splits(
                source,
                env_root,
                seeds,
                overwrite=False,
            )

            self.assertEqual(paths, repeated)
            self.assertEqual(len(paths), 5)
            self.assertTrue((env_root / "real_pool.npz").is_file())
            self.assertTrue(all(path.is_file() for path in paths))
            with np.load(paths[0], allow_pickle=False) as saved:
                self.assertEqual(saved["val_traj"].shape[0], 600)
                self.assertEqual(saved["test_traj"].shape[0], 200)


class MetricTests(unittest.TestCase):
    def test_aggregate_uses_population_standard_deviation(self):
        aggregate = rte.aggregate_values([90.0, 92.0, 94.0, 96.0, 98.0])

        self.assertEqual(aggregate["count"], 5)
        self.assertEqual(aggregate["ddof"], 0)
        self.assertAlmostEqual(aggregate["mean"], 94.0)
        self.assertAlmostEqual(aggregate["std"], np.std([90, 92, 94, 96, 98], ddof=0))

    def test_symbolic_raw_summary_has_no_repeat_standard_deviation(self):
        summary = rte.single_result_summary(coverage=0.4275, average_area=0.001293)

        self.assertEqual(summary["coverage_mean"], 0.4275)
        self.assertIsNone(summary["coverage_std"])
        self.assertEqual(summary["area_mean"], 0.001293)
        self.assertIsNone(summary["area_std"])
        self.assertEqual(summary["num_repeats"], 1)

    def test_strict_metrics_reject_an_invalid_trajectory_instead_of_dropping_it(self):
        grid = rte.stm.GridInfo(
            names=["x", "y"],
            starts=np.array([0.0, 0.0]),
            stops=np.array([1.0, 1.0]),
            nums=np.array([1, 1]),
            steps=np.array([1.0, 1.0]),
        )
        cells = [{"bounds": [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]}]
        trajectories = np.array([[[2.0, 2.0], [2.0, 2.0]]])

        with self.assertRaisesRegex(ValueError, "every trajectory"):
            rte.strict_table_metrics(trajectories, grid, cells, dims=(0, 1))

    def test_inflated_repeat_calibrates_gamma_on_val_and_evaluates_test(self):
        grid = rte.stm.GridInfo(
            names=["x", "y"],
            starts=np.array([0.0, 0.0]),
            stops=np.array([1.0, 1.0]),
            nums=np.array([1, 1]),
            steps=np.array([1.0, 1.0]),
        )
        cells = [{"bounds": [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]}]
        val = np.full((600, 2, 2), 0.5)
        val[:, 1, 0] = 1.2
        test = np.full((200, 2, 2), 0.5)
        test[:, 1, 0] = 1.15

        result = rte.evaluate_inflated_repeat(
            val,
            test,
            grid,
            cells,
            dims=(0, 1),
            alpha=0.05,
        )

        self.assertAlmostEqual(result["gamma"], 0.2)
        self.assertAlmostEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(result["average_area"], 1.96)
        self.assertEqual(result["calibration_size"], 600)
        self.assertEqual(result["evaluation_size"], 200)

    def test_five_repeat_metrics_are_aggregated_across_repeats(self):
        rows = [
            {"coverage": value / 100.0, "average_area": value / 1000.0}
            for value in (90, 92, 94, 96, 98)
        ]

        aggregate = rte.aggregate_metric_rows(rows)

        self.assertAlmostEqual(aggregate["coverage_mean"], 0.94)
        self.assertAlmostEqual(
            aggregate["coverage_std"],
            np.std([0.90, 0.92, 0.94, 0.96, 0.98], ddof=0),
        )
        self.assertAlmostEqual(aggregate["area_mean"], 0.094)
        self.assertAlmostEqual(
            aggregate["area_std"],
            np.std([0.090, 0.092, 0.094, 0.096, 0.098], ddof=0),
        )
        self.assertEqual(aggregate["num_repeats"], 5)
        self.assertEqual(aggregate["std_ddof"], 0)


class PredictorEvaluationTests(unittest.TestCase):
    def test_cli_exposes_predictor_evaluation_command(self):
        args = rte.build_parser().parse_args(["evaluate-predictor"])

        self.assertEqual(args.command, "evaluate-predictor")

    def test_predictor_paths_are_the_five_existing_shared_tubes(self):
        paths = rte.predictor_tube_paths("cartpole")

        self.assertEqual(
            [path.name for path in paths],
            [f"predictor_tube_seed_{index}.json" for index in range(5)],
        )
        self.assertTrue(all(path.is_file() for path in paths))

    def test_predictor_case_pairs_existing_tubes_with_five_splits(self):
        tube = {
            "method": "transformer_sampled_minmax_envelope",
            "grid": {"dims": [
                {"name": "x", "start": 0.0, "stop": 1.0, "num": 1},
                {"name": "y", "start": 0.0, "stop": 1.0, "num": 1},
            ]},
            "cells": [{"bounds": [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]}],
        }
        split = {
            "val_traj": np.full((600, 2, 2), 0.5),
            "test_traj": np.full((200, 2, 2), 0.5),
        }
        repeat_seeds = rte.derive_repeat_seeds(2025)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(5):
                path = root / f"predictor_tube_seed_{index}.json"
                path.write_text(json.dumps(tube))
                paths.append(path)
            spec = {
                "dims": (0, 1),
                "expected_cells": 1,
                "expected_horizon": 2,
            }
            with (
                mock.patch.dict(rte.ENVIRONMENTS, {"toy": spec}),
                mock.patch.object(rte, "load_pool_artifact", return_value=(None, "pool")),
                mock.patch.object(rte, "load_repeat_splits", return_value=[split] * 5),
            ):
                aggregate = rte.evaluate_predictor_case(
                    "toy",
                    paths,
                    alpha=0.05,
                    data_root=root / "data",
                    result_root=root / "results",
                    repeat_seeds=repeat_seeds,
                    overwrite=False,
                )

            self.assertEqual(aggregate["construction"], "predictor")
            self.assertEqual(aggregate["raw"]["num_repeats"], 5)
            self.assertEqual(aggregate["raw"]["std_ddof"], 0)
            self.assertEqual(aggregate["inflated"]["std_ddof"], 0)
            for index in range(5):
                repeat_dir = root / "results" / "toy" / "c_predictor" / f"repeat_{index:02d}"
                self.assertTrue((repeat_dir / "raw_metrics.json").is_file())
                self.assertTrue((repeat_dir / "calibration.json").is_file())
                self.assertTrue((repeat_dir / "inflated_metrics.json").is_file())

    def test_predictor_summary_writes_raw_and_formatted_csv(self):
        aggregate = {
            "raw": {
                "coverage_mean": 0.1,
                "coverage_std": 0.02,
                "area_mean": 0.001,
                "area_std": 0.0001,
                "num_repeats": 5,
                "std_ddof": 0,
            },
            "inflated": {
                "coverage_mean": 0.95,
                "coverage_std": 0.01,
                "area_mean": 0.002,
                "area_std": 0.0002,
                "num_repeats": 5,
                "std_ddof": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "toy" / "c_predictor" / "aggregate.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(aggregate))

            rte.write_predictor_summary(["toy"], root, overwrite=False)

            raw = (root / "predictor_metrics_raw.csv").read_text()
            formatted = (root / "predictor_metrics_formatted.csv").read_text()
            self.assertIn("TP", raw)
            self.assertIn("TP (inflate)", raw)
            self.assertIn("10.00 ± 2.00", formatted)
            self.assertIn("95.00 ± 1.00", formatted)


class UnifiedSamplingConfigTests(unittest.TestCase):
    def test_materialized_dwm_and_cgan_configs_reference_one_canonical_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwm_path = rte.materialize_case_configs(
                "cartpole", "dwm", result_root=root, overwrite=False
            )
            cgan_path = rte.materialize_case_configs(
                "cartpole", "cgan", result_root=root, overwrite=False
            )
            dwm = json.loads(dwm_path.read_text())
            cgan = json.loads(cgan_path.read_text())

        self.assertEqual(dwm["grid_config"], cgan["grid_config"])

    def test_experiment_overrides_fix_brake_grid_and_cartpole_cgan_latent_range(self):
        brake_base = {"decoder": {"name": "Decoder"}}
        brake_starv = {
            "grid": {"dims": [
                {"name": "dis", "start": 6.0, "stop": 6.4, "num": 40},
                {"name": "vel", "start": 6.0, "stop": 6.04, "num": 40},
            ]}
        }
        brake_grid = json.loads(json.dumps(brake_starv))
        _, adjusted_starv, adjusted_grid = rte.apply_experiment_overrides(
            "brake_system", "dwm", brake_base, brake_starv, brake_grid
        )
        self.assertEqual(adjusted_starv["grid"]["dims"][1]["stop"], 6.4)
        self.assertEqual(adjusted_grid["grid"]["dims"][1]["stop"], 6.4)
        self.assertEqual(
            np.prod([dim["num"] for dim in adjusted_grid["grid"]["dims"]]),
            1600,
        )

        cart_base = {"decoder": {"name": "G_MLP", "args": {"z_range": 0.8}}}
        cart_starv = {
            "layers": {"G_MLP": {"kwargs": {"z_range": 0.8}}},
            "grid": {"dims": []},
        }
        adjusted_base, adjusted_starv, _ = rte.apply_experiment_overrides(
            "cartpole", "cgan", cart_base, cart_starv, {"grid": {"dims": []}}
        )
        self.assertEqual(adjusted_base["decoder"]["args"]["z_range"], 0.05)
        self.assertEqual(adjusted_starv["layers"]["G_MLP"]["kwargs"]["z_range"], 0.05)

    def test_repeat_config_redirects_all_generated_files_to_experiment_tree(self):
        config = {
            "grid_config": "config/starv_verification/cartpole.json",
            "model_id": "saliency",
            "samples_per_cell": 3,
            "seed": 728,
            "cell_batch_size": 128,
            "states_file": "old/states.npz",
            "trajectory_file": "old/trajectories.npz",
            "tube_file": "old/tube.json",
        }

        actual = rte.build_repeat_sampling_config(
            config,
            env="cartpole",
            decoder="dwm",
            repeat_index=2,
            tube_seed=456,
            data_root=Path("datasets/repeated"),
            result_root=Path("results/repeated"),
        )

        self.assertEqual(actual["seed"], 456)
        self.assertEqual(
            actual["states_file"],
            "datasets/repeated/cartpole/sampled/repeat_02/cellwise_states.npz",
        )
        self.assertEqual(
            actual["trajectory_file"],
            "datasets/repeated/cartpole/sampled/repeat_02/sampled_trajectories_saliency.npz",
        )
        self.assertEqual(
            actual["tube_file"],
            "results/repeated/cartpole/b_sampled/dwm/repeat_02/sampled_tube.json",
        )

    def test_dwm_and_cgan_repeat_configs_share_the_state_file(self):
        base = {
            "grid_config": "grid.json",
            "samples_per_cell": 3,
            "seed": 728,
            "cell_batch_size": 128,
            "states_file": "old/states.npz",
            "trajectory_file": "old/trajectories.npz",
            "tube_file": "old/tube.json",
        }
        dwm = rte.build_repeat_sampling_config(
            {**base, "model_id": "old"},
            env="brake_system",
            decoder="dwm",
            repeat_index=1,
            tube_seed=789,
            data_root=Path("datasets/repeated"),
            result_root=Path("results/repeated"),
        )
        cgan = rte.build_repeat_sampling_config(
            {**base, "model_id": "g_mlp"},
            env="brake_system",
            decoder="cgan",
            repeat_index=1,
            tube_seed=789,
            data_root=Path("datasets/repeated"),
            result_root=Path("results/repeated"),
        )

        self.assertEqual(dwm["states_file"], cgan["states_file"])
        self.assertNotEqual(dwm["trajectory_file"], cgan["trajectory_file"])
        self.assertNotEqual(dwm["tube_file"], cgan["tube_file"])


if __name__ == "__main__":
    unittest.main()
