import copy
import importlib
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "verifiable-wm-matplotlib"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GridCellTests(unittest.TestCase):
    def test_grid_cell_bounds_match_starv_ij_c_order(self):
        try:
            sampled_tube = importlib.import_module("sampled_tube")
        except ModuleNotFoundError:
            self.fail("sampled_tube module is required")

        grid = {
            "dims": [
                {"name": "x", "start": 0.0, "stop": 2.0, "num": 2},
                {"name": "y", "start": -1.0, "stop": 1.0, "num": 2},
            ]
        }

        actual = sampled_tube.grid_cell_bounds(grid)
        expected = np.array(
            [
                [[0.0, 1.0], [-1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
                [[1.0, 2.0], [-1.0, 0.0]],
                [[1.0, 2.0], [0.0, 1.0]],
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, expected)

    def test_invalid_grid_dimensions_are_rejected(self):
        sampled_tube = importlib.import_module("sampled_tube")
        cases = (
            (
                {
                    "dims": [
                        {"name": "x", "start": 0.0, "stop": 1.0, "num": 0}
                    ]
                },
                "positive",
            ),
            (
                {
                    "dims": [
                        {"name": "x", "start": 1.0, "stop": 0.0, "num": 1}
                    ]
                },
                "start",
            ),
            (
                {
                    "dims": [
                        {
                            "name": "x",
                            "start": 0.0,
                            "stop": np.inf,
                            "num": 1,
                        }
                    ]
                },
                "finite",
            ),
        )
        for grid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    sampled_tube.grid_cell_bounds(grid)


class CellwiseRandomTests(unittest.TestCase):
    def setUp(self):
        self.sampled_tube = importlib.import_module("sampled_tube")
        self.cells = np.array(
            [
                [[0.0, 1.0], [2.0, 2.0]],
                [[1.0, 2.0], [-1.0, 1.0]],
                [[2.0, 3.0], [5.0, 6.0]],
            ],
            dtype=np.float64,
        )

    def test_states_are_three_per_cell_inside_and_reproducible(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "sample_cellwise_states"),
            "sample_cellwise_states is required",
        )

        first = self.sampled_tube.sample_cellwise_states(
            self.cells, samples_per_cell=3, seed=2025
        )
        second = self.sampled_tube.sample_cellwise_states(
            self.cells, samples_per_cell=3, seed=2025
        )

        self.assertEqual(first.shape, (3, 3, 2))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first >= self.cells[:, None, :, 0]))
        self.assertTrue(np.all(first <= self.cells[:, None, :, 1]))
        np.testing.assert_array_equal(first[:, :, 1][0], np.full(3, 2.0))

    def test_state_stream_for_existing_cells_does_not_depend_on_cell_count(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "sample_cellwise_states"),
            "sample_cellwise_states is required",
        )

        prefix = self.sampled_tube.sample_cellwise_states(
            self.cells[:2], samples_per_cell=3, seed=2025
        )
        full = self.sampled_tube.sample_cellwise_states(
            self.cells, samples_per_cell=3, seed=2025
        )

        np.testing.assert_array_equal(prefix, full[:2])

    def test_latents_are_float32_reproducible_and_in_range(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "sample_cellwise_latents"),
            "sample_cellwise_latents is required",
        )

        latents = self.sampled_tube.sample_cellwise_latents(
            num_cells=3,
            samples_per_cell=3,
            num_steps=4,
            latent_dim=2,
            z_range=0.8,
            seed=2025,
        )
        repeated = self.sampled_tube.sample_cellwise_latents(
            num_cells=3,
            samples_per_cell=3,
            num_steps=4,
            latent_dim=2,
            z_range=0.8,
            seed=2025,
        )

        self.assertEqual(latents.shape, (3, 3, 4, 2))
        self.assertEqual(latents.dtype, np.float32)
        np.testing.assert_array_equal(latents, repeated)
        self.assertTrue(np.all(latents >= -0.8))
        self.assertTrue(np.all(latents <= 0.8))

    def test_invalid_cell_bounds_are_rejected_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.sampled_tube.sample_cellwise_states(
                np.zeros((1, 2, 3)),
                samples_per_cell=3,
                seed=2025,
            )

        invalid_order = self.cells.copy()
        invalid_order[0, 0] = [1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "lower"):
            self.sampled_tube.sample_cellwise_states(
                invalid_order,
                samples_per_cell=3,
                seed=2025,
            )

        nonfinite = self.cells.copy()
        nonfinite[0, 0, 1] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            self.sampled_tube.sample_cellwise_states(
                nonfinite,
                samples_per_cell=3,
                seed=2025,
            )


class SampledTubeJsonTests(unittest.TestCase):
    def setUp(self):
        self.sampled_tube = importlib.import_module("sampled_tube")

    def test_cells_use_full_initial_cell_then_sample_envelope_and_result(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "sampled_cells_from_trajectories"),
            "sampled_cells_from_trajectories is required",
        )
        cell_bounds = np.array(
            [
                [[0.0, 1.0], [-1.0, 1.0]],
                [[1.0, 2.0], [-1.0, 1.0]],
            ],
            dtype=np.float64,
        )
        trajectories = np.array(
            [
                [
                    [[0.1, 0.0], [0.5, -0.1], [0.61, 0.0]],
                    [[0.4, 0.2], [0.6, 0.0], [0.70, 0.1]],
                    [[0.9, -0.2], [0.7, 0.1], [0.80, -0.1]],
                ],
                [
                    [[1.1, 0.0], [0.4, -0.2], [0.59, 0.0]],
                    [[1.4, 0.2], [0.5, 0.0], [0.70, 0.1]],
                    [[1.9, -0.2], [0.6, 0.2], [0.80, -0.1]],
                ],
            ],
            dtype=np.float32,
        )
        verifier = {
            "name": "MountainCarVerifier",
            "kwargs": {"goal_position_threshold": 0.6, "num_steps": 2},
        }

        cells = self.sampled_tube.sampled_cells_from_trajectories(
            trajectories, cell_bounds, verifier
        )

        self.assertEqual(cells[0]["bounds"][0], [[0.0, 1.0], [-1.0, 1.0]])
        np.testing.assert_allclose(
            cells[0]["bounds"][1],
            [[0.5, 0.7], [-0.1, 0.1]],
        )
        self.assertIs(cells[0]["result"], True)
        self.assertIs(cells[1]["result"], False)

    def test_pendulum_wrap_uses_flat_interval_pairs(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "sampled_cells_from_trajectories"),
            "sampled_cells_from_trajectories is required",
        )
        cell_bounds = np.array(
            [[[-np.pi, np.pi], [-1.0, 1.0]]],
            dtype=np.float64,
        )
        trajectories = np.array(
            [
                [
                    [[3.0, 0.0], [3.10, -0.2]],
                    [[-3.0, 0.0], [-3.10, 0.0]],
                    [[3.1, 0.0], [3.12, 0.2]],
                ]
            ],
            dtype=np.float32,
        )
        verifier = {
            "name": "PendulumVerifier",
            "kwargs": {"goal_angle_threshold": 0.2, "num_steps": 1},
        }

        cells = self.sampled_tube.sampled_cells_from_trajectories(
            trajectories, cell_bounds, verifier
        )
        theta_bounds = cells[0]["bounds"][1][0]

        self.assertEqual(len(theta_bounds), 4)
        np.testing.assert_allclose(
            theta_bounds,
            [3.10, np.pi, -np.pi, -3.10],
            atol=1e-6,
        )

    def test_written_result_is_consumed_by_existing_compare(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "build_sampled_safety_result"),
            "build_sampled_safety_result is required",
        )
        self.assertTrue(
            hasattr(self.sampled_tube, "write_json_atomic"),
            "write_json_atomic is required",
        )
        self.assertIn(
            "grid_config",
            inspect.signature(
                self.sampled_tube.build_sampled_safety_result
            ).parameters,
            "build_sampled_safety_result must receive the canonical grid config",
        )
        compare = importlib.import_module("compare")
        stm = importlib.import_module("signed_tube_margin")
        source = {
            "layers": {"Decoder": {"args": [], "kwargs": {"weights": "x.pth"}}},
            "verifier": {
                "name": "MountainCarVerifier",
                "args": [],
                "kwargs": {
                    "goal_position_threshold": 0.6,
                    "num_steps": 1,
                    "early_stop": False,
                },
            },
            "grid": {
                "dims": [
                    {"name": "pos", "start": 0.0, "stop": 1.0, "num": 1},
                    {"name": "vel", "start": -1.0, "stop": 1.0, "num": 1},
                ]
            },
            "output_prefix": "unused",
        }
        canonical = copy.deepcopy(source)
        canonical["grid"]["dims"][0]["name"] = "canonical_pos"
        cells = [
            {
                "cell_index": 0,
                "bounds": [
                    [[0.0, 1.0], [-1.0, 1.0]],
                    [[0.6, 0.8], [-0.1, 0.1]],
                ],
                "result": True,
            }
        ]
        payload = self.sampled_tube.build_sampled_safety_result(
            source,
            cells,
            {
                "model_id": "saliency",
                "samples_per_cell": 3,
                "seed": 2025,
                "horizon": 1,
                "state_dim": 2,
                "grid_fingerprint": self.sampled_tube.grid_fingerprint(
                    canonical["grid"]
                ),
            },
            grid_config=canonical,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sampled.json"
            self.sampled_tube.write_json_atomic(path, payload)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            grid, loaded_cells = stm.load_safety_result(path)

        self.assertEqual(
            loaded["method"], "cellwise_sampled_trajectory_envelope"
        )
        self.assertEqual(loaded["grid"]["dims"][0]["name"], "canonical_pos")
        # Importing the StarV/codac stack leaves the FPU in a rounding mode
        # that shifts JSON float parsing by 1 ulp, so the round-tripped
        # bounds are compared numerically instead of by exact equality.
        self.assertEqual(len(loaded_cells), len(cells))
        for loaded_cell, expected_cell in zip(loaded_cells, cells):
            self.assertEqual(loaded_cell["cell_index"], expected_cell["cell_index"])
            self.assertEqual(loaded_cell["result"], expected_cell["result"])
            np.testing.assert_allclose(
                np.asarray(loaded_cell["bounds"], dtype=float),
                np.asarray(expected_cell["bounds"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        old_check_dims = compare.CHECK_DIMS
        old_max_steps = compare.MAX_STEPS
        old_delta = compare.DELTA
        evaluation = np.array(
            [[[0.2, 0.0], [0.7, 0.0]]],
            dtype=np.float32,
        )
        try:
            compare.CHECK_DIMS = (0, 1)
            compare.MAX_STEPS = None
            compare.DELTA = 0.0
            rows = compare.compare_set(
                evaluation,
                grid,
                loaded_cells,
            )
        finally:
            compare.CHECK_DIMS = old_check_dims
            compare.MAX_STEPS = old_max_steps
            compare.DELTA = old_delta

        self.assertTrue(rows[0]["fully_inside"])

    def test_writer_rejects_a_truncated_sampled_tube(self):
        payload = {
            "method": "cellwise_sampled_trajectory_envelope",
            "guarantee_type": "empirical_only",
            "layers": {"Decoder": {"args": [], "kwargs": {}}},
            "verifier": {
                "name": "MountainCarVerifier",
                "kwargs": {"num_steps": 1},
            },
            "grid": {
                "dims": [
                    {"name": "x", "start": 0.0, "stop": 1.0, "num": 1},
                    {"name": "v", "start": 0.0, "stop": 0.0, "num": 1},
                ]
            },
            "cells": [
                {
                    "cell_index": 0,
                    "bounds": [[[0.0, 1.0], [0.0, 0.0]]],
                    "result": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "horizon"):
                self.sampled_tube.write_json_atomic(
                    Path(tmp) / "truncated.json", payload
                )

    def test_writer_rejects_inconsistent_sampled_tube_metadata(self):
        grid = {
            "dims": [
                {"name": "x", "start": 0.0, "stop": 1.0, "num": 1},
                {"name": "v", "start": 0.0, "stop": 0.0, "num": 1},
            ]
        }
        payload = {
            "method": "cellwise_sampled_trajectory_envelope",
            "guarantee_type": "empirical_only",
            "samples_per_cell": 3,
            "seed": 2025,
            "horizon": 1,
            "state_dim": 2,
            "grid_fingerprint": self.sampled_tube.grid_fingerprint(grid),
            "layers": {"Decoder": {"args": [], "kwargs": {}}},
            "verifier": {
                "name": "MountainCarVerifier",
                "kwargs": {"num_steps": 1},
            },
            "grid": grid,
            "cells": [
                {
                    "cell_index": 0,
                    "bounds": [
                        [[0.0, 1.0], [0.0, 0.0]],
                        [[0.2, 0.8], [0.0, 0.0]],
                    ],
                    "result": True,
                }
            ],
        }

        corruptions = (
            ("guarantee_type", "formal", "guarantee_type"),
            ("samples_per_cell", 2, "samples_per_cell"),
            ("horizon", 2, "horizon metadata"),
            ("state_dim", 3, "state_dim metadata"),
            ("grid_fingerprint", "wrong", "fingerprint"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for key, value, message in corruptions:
                with self.subTest(key=key):
                    invalid = copy.deepcopy(payload)
                    invalid[key] = value
                    with self.assertRaisesRegex(ValueError, message):
                        self.sampled_tube.write_json_atomic(
                            Path(tmp) / f"{key}.json", invalid
                        )

    def test_invalid_trajectory_protocol_is_rejected_before_json(self):
        cell_bounds = np.array(
            [
                [[0.0, 1.0], [-1.0, 1.0]],
                [[1.0, 2.0], [-1.0, 1.0]],
            ],
            dtype=np.float64,
        )
        verifier = {
            "name": "MountainCarVerifier",
            "kwargs": {"goal_position_threshold": 0.6, "num_steps": 2},
        }
        valid = np.zeros((2, 3, 3, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "cell count"):
            self.sampled_tube.sampled_cells_from_trajectories(
                valid[:1], cell_bounds, verifier
            )
        with self.assertRaisesRegex(ValueError, "samples per cell"):
            self.sampled_tube.sampled_cells_from_trajectories(
                valid[:, :2], cell_bounds, verifier
            )
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.sampled_tube.sampled_cells_from_trajectories(
                valid[:, :, :2], cell_bounds, verifier
            )
        invalid = valid.copy()
        invalid[0, 0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN|Inf|finite"):
            self.sampled_tube.sampled_cells_from_trajectories(
                invalid, cell_bounds, verifier
            )

    def test_cartpole_pendulum_and_brake_result_rules(self):
        cases = [
            (
                "CartpoleVerifier",
                {"goal_angle_threshold": 0.2, "num_steps": 1},
                np.array(
                    [
                        [
                            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.10, 0.0]],
                            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.19, 0.0]],
                            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.05, 0.0]],
                        ]
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [[[-1.0, 1.0], [0.0, 0.0], [-0.2, 0.2], [0.0, 0.0]]]
                ),
                True,
            ),
            (
                "PendulumVerifier",
                {"goal_angle_threshold": 0.2, "num_steps": 1},
                np.array(
                    [
                        [
                            [[0.0, 0.0], [0.21, 0.0]],
                            [[0.0, 0.0], [0.00, 0.0]],
                            [[0.0, 0.0], [-0.10, 0.0]],
                        ]
                    ],
                    dtype=np.float32,
                ),
                np.array([[[-np.pi, np.pi], [-1.0, 1.0]]]),
                False,
            ),
            (
                "BrakeVerifier",
                {"num_steps": 1},
                np.array(
                    [
                        [
                            [[1.0, 1.0], [0.0, 0.9]],
                            [[1.1, 1.0], [0.1, 0.9]],
                            [[1.2, 1.0], [0.2, 0.9]],
                        ]
                    ],
                    dtype=np.float32,
                ),
                np.array([[[1.0, 2.0], [1.0, 1.0]]]),
                False,
            ),
        ]

        for name, kwargs, trajectories, bounds, expected in cases:
            with self.subTest(verifier=name):
                cells = self.sampled_tube.sampled_cells_from_trajectories(
                    trajectories,
                    bounds,
                    {"name": name, "kwargs": kwargs},
                )
                self.assertIs(cells[0]["result"], expected)


class CellwiseConfigTests(unittest.TestCase):
    def setUp(self):
        self.sampled_tube = importlib.import_module("sampled_tube")

    @staticmethod
    def load(relative_path):
        return json.loads((PROJECT_ROOT / relative_path).read_text())

    def test_existing_dwm_and_g_mlp_configs_validate_against_shared_grid(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "validate_cellwise_setup"),
            "validate_cellwise_setup is required",
        )
        canonical = self.load("config/starv_verification/cartpole.json")

        cases = [
            ("config/sampling/cartpole.json", "saliency", False),
            ("config/sampling/cartpole_g_mlp.json", "g_mlp", True),
        ]
        for base_path, model_id, formal_match in cases:
            with self.subTest(model_id=model_id):
                base = self.load(base_path)
                model_starv = self.load(base["starv_config"])
                wrapper = {
                    "sampling_mode": "cellwise_tube",
                    "model_id": model_id,
                    "samples_per_cell": 3,
                    "seed": 2025,
                    "cell_batch_size": 128,
                }
                metadata = self.sampled_tube.validate_cellwise_setup(
                    wrapper, base, model_starv, canonical
                )
                self.assertEqual(metadata["model_id"], model_id)
                self.assertIs(
                    metadata["formal_semantics_match"], formal_match
                )

    def test_grid_mismatch_is_rejected(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "validate_cellwise_setup"),
            "validate_cellwise_setup is required",
        )
        base = self.load("config/sampling/cartpole.json")
        model_starv = self.load(base["starv_config"])
        canonical = copy.deepcopy(model_starv)
        canonical["grid"]["dims"][0]["num"] += 1
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "saliency",
            "samples_per_cell": 3,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        with self.assertRaisesRegex(ValueError, "grid"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, model_starv, canonical
            )

    def test_fixed_three_samples_and_positive_batch_are_enforced_preflight(self):
        base = self.load("config/sampling/cartpole.json")
        model_starv = self.load(base["starv_config"])
        canonical = copy.deepcopy(model_starv)
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "saliency",
            "samples_per_cell": 2,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        with self.assertRaisesRegex(ValueError, "exactly 3"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, model_starv, canonical
            )

        wrapper["samples_per_cell"] = 3
        wrapper["cell_batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "cell_batch_size"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, model_starv, canonical
            )

    def test_cartpole_decoder_state_projection_must_match_verifier(self):
        base = self.load("config/sampling/cartpole_g_mlp.json")
        model_starv = self.load(base["starv_config"])
        canonical = self.load("config/starv_verification/cartpole.json")
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "g_mlp",
            "samples_per_cell": 3,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        metadata = self.sampled_tube.validate_cellwise_setup(
            wrapper, base, model_starv, canonical
        )
        self.assertEqual(metadata["decoder_state_indices"], [0, 2])

        base["decoder_state_indices"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "decoder_state_indices"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, model_starv, canonical
            )

    def test_dynamic_parameters_must_match_fixed_verifier_dynamics(self):
        base = self.load("config/sampling/brake_system_g_mlp.json")
        model_starv = self.load(base["starv_config"])
        canonical = self.load("config/starv_verification/brake_system.json")
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "g_mlp",
            "samples_per_cell": 3,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        base["dynamic"]["args"] = {"dt": 0.2}
        with self.assertRaisesRegex(ValueError, "dynamic"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, model_starv, canonical
            )

    def test_brake_decoder_requires_clamp_output(self):
        base = self.load("config/sampling/brake_system.json")
        model_starv = self.load(base["starv_config"])
        canonical = copy.deepcopy(model_starv)
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "old",
            "samples_per_cell": 3,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        metadata = self.sampled_tube.validate_cellwise_setup(
            wrapper, base, model_starv, canonical
        )
        self.assertIs(metadata["formal_semantics_match"], True)

        base["decoder"]["args"]["output_activation"] = "sigmoid"
        with self.assertRaisesRegex(ValueError, "output activation"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, model_starv, canonical
            )

    def test_early_stop_and_starv_layer_order_are_rejected(self):
        base = self.load("config/sampling/mountain_car_g_mlp.json")
        model_starv = self.load(base["starv_config"])
        canonical = self.load("config/starv_verification/mountain_car.json")
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "g_mlp",
            "samples_per_cell": 3,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        early_model = copy.deepcopy(model_starv)
        early_grid = copy.deepcopy(canonical)
        early_model["verifier"]["kwargs"]["early_stop"] = True
        early_grid["verifier"]["kwargs"]["early_stop"] = True
        with self.assertRaisesRegex(ValueError, "early_stop"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, early_model, early_grid
            )

        reordered = copy.deepcopy(model_starv)
        reordered["layers"] = {
            "Controller": reordered["layers"]["Controller"],
            "G_MLP": reordered["layers"]["G_MLP"],
        }
        with self.assertRaisesRegex(ValueError, "layer order"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, base, reordered, canonical
            )

    def test_g_mlp_architecture_args_must_match_supported_forward(self):
        base = self.load("config/sampling/pendulum_g_mlp.json")
        model_starv = self.load(base["starv_config"])
        canonical = self.load("config/starv_verification/pendulum.json")
        wrapper = {
            "sampling_mode": "cellwise_tube",
            "model_id": "g_mlp",
            "samples_per_cell": 3,
            "seed": 2025,
            "cell_batch_size": 128,
        }

        bad_output_dim = copy.deepcopy(base)
        bad_output_dim["decoder"]["args"]["output_dim"] = 10
        bad_starv = copy.deepcopy(model_starv)
        bad_starv["layers"]["G_MLP"]["kwargs"]["output_dim"] = 10
        with self.assertRaisesRegex(ValueError, "output_dim"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, bad_output_dim, bad_starv, canonical
            )

        bad_activation = copy.deepcopy(base)
        bad_activation["decoder"]["args"]["output_activation"] = "clamp"
        with self.assertRaisesRegex(ValueError, "output_activation"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper, bad_activation, model_starv, canonical
            )

        bad_starv_activation = copy.deepcopy(model_starv)
        bad_starv_activation["layers"]["G_MLP"]["kwargs"][
            "output_activation"
        ] = "clamp"
        with self.assertRaisesRegex(ValueError, "output_activation"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper,
                base,
                bad_starv_activation,
                canonical,
            )

        bad_controller = copy.deepcopy(base)
        bad_controller["controller"]["args"]["activation"] = "relu"
        bad_controller_starv = copy.deepcopy(model_starv)
        bad_controller_starv["layers"]["Controller"]["kwargs"][
            "activation"
        ] = "relu"
        with self.assertRaisesRegex(ValueError, "controller activation"):
            self.sampled_tube.validate_cellwise_setup(
                wrapper,
                bad_controller,
                bad_controller_starv,
                canonical,
            )


class CellwiseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.sampled_tube = importlib.import_module("sampled_tube")
        self.grid = {
            "dims": [
                {"name": "x", "start": 0.0, "stop": 2.0, "num": 2},
                {"name": "fixed", "start": 1.0, "stop": 1.0, "num": 1},
            ]
        }
        self.cells = self.sampled_tube.grid_cell_bounds(self.grid)

    def test_shared_states_are_created_then_reused(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "load_or_create_cellwise_states"),
            "load_or_create_cellwise_states is required",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cellwise_states.npz"
            first = self.sampled_tube.load_or_create_cellwise_states(
                path,
                self.cells,
                samples_per_cell=3,
                seed=2025,
                grid=self.grid,
                horizon=2,
                source_grid_config="canonical.json",
            )
            second = self.sampled_tube.load_or_create_cellwise_states(
                path,
                self.cells,
                samples_per_cell=3,
                seed=2025,
                grid=self.grid,
                horizon=2,
                source_grid_config="canonical.json",
            )

            np.testing.assert_array_equal(first, second)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["states"].shape, (2, 3, 2))
                self.assertEqual(int(saved["seed"]), 2025)
                self.assertEqual(int(saved["samples_per_cell"]), 3)

    def test_shared_states_metadata_mismatch_is_rejected(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "load_or_create_cellwise_states"),
            "load_or_create_cellwise_states is required",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cellwise_states.npz"
            self.sampled_tube.load_or_create_cellwise_states(
                path,
                self.cells,
                samples_per_cell=3,
                seed=2025,
                grid=self.grid,
                horizon=2,
                source_grid_config="canonical.json",
            )

            with self.assertRaisesRegex(ValueError, "seed"):
                self.sampled_tube.load_or_create_cellwise_states(
                    path,
                    self.cells,
                    samples_per_cell=3,
                    seed=7,
                    grid=self.grid,
                    horizon=2,
                    source_grid_config="canonical.json",
                )

    def test_shared_states_cell_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cellwise_states.npz"
            self.sampled_tube.load_or_create_cellwise_states(
                path,
                self.cells,
                samples_per_cell=3,
                seed=2025,
                grid=self.grid,
                horizon=2,
                source_grid_config="canonical.json",
            )
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: np.asarray(saved[key]) for key in saved.files}
            arrays["cell_indices"] = np.array([1, 0], dtype=np.int64)
            self.sampled_tube.write_npz_atomic(path, **arrays)

            with self.assertRaisesRegex(ValueError, "cell_indices"):
                self.sampled_tube.load_or_create_cellwise_states(
                    path,
                    self.cells,
                    samples_per_cell=3,
                    seed=2025,
                    grid=self.grid,
                    horizon=2,
                    source_grid_config="canonical.json",
                )

    def test_shared_states_source_dtype_and_seed_values_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cellwise_states.npz"
            expected = self.sampled_tube.load_or_create_cellwise_states(
                path,
                self.cells,
                samples_per_cell=3,
                seed=2025,
                grid=self.grid,
                horizon=2,
                source_grid_config="canonical.json",
            )
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: np.asarray(saved[key]) for key in saved.files}

            arrays["source_grid_config"] = np.array("different.json")
            self.sampled_tube.write_npz_atomic(path, **arrays)
            with self.assertRaisesRegex(ValueError, "source_grid_config"):
                self.sampled_tube.load_or_create_cellwise_states(
                    path,
                    self.cells,
                    samples_per_cell=3,
                    seed=2025,
                    grid=self.grid,
                    horizon=2,
                    source_grid_config="canonical.json",
                )

            arrays["source_grid_config"] = np.array("canonical.json")
            arrays["states"] = expected.astype(np.float64)
            self.sampled_tube.write_npz_atomic(path, **arrays)
            with self.assertRaisesRegex(ValueError, "float32"):
                self.sampled_tube.load_or_create_cellwise_states(
                    path,
                    self.cells,
                    samples_per_cell=3,
                    seed=2025,
                    grid=self.grid,
                    horizon=2,
                    source_grid_config="canonical.json",
                )

            arrays["states"] = expected.copy()
            arrays["states"][0, 0, 0] = (
                arrays["states"][0, 0, 0] + np.float32(1e-4)
            )
            self.sampled_tube.write_npz_atomic(path, **arrays)
            with self.assertRaisesRegex(ValueError, "fixed seed"):
                self.sampled_tube.load_or_create_cellwise_states(
                    path,
                    self.cells,
                    samples_per_cell=3,
                    seed=2025,
                    grid=self.grid,
                    horizon=2,
                    source_grid_config="canonical.json",
                )

    def test_npz_writer_round_trips_without_partial_suffix(self):
        self.assertTrue(
            hasattr(self.sampled_tube, "write_npz_atomic"),
            "write_npz_atomic is required",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectories.npz"
            self.sampled_tube.write_npz_atomic(
                path,
                trajectories=np.ones((1, 3, 2, 2), dtype=np.float32),
                model_id=np.array("saliency"),
            )

            self.assertTrue(path.exists())
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["trajectories"].shape, (1, 3, 2, 2))
                self.assertEqual(str(saved["model_id"]), "saliency")

    def test_npz_writer_rejects_object_arrays_without_final_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.npz"
            with self.assertRaisesRegex(ValueError, "object"):
                self.sampled_tube.write_npz_atomic(
                    path,
                    invalid=np.array([{"not": "portable"}], dtype=object),
                )
            self.assertFalse(path.exists())

    def test_rollout_artifact_rejects_nonfinite_or_out_of_range_arrays(self):
        trajectories = np.zeros((2, 3, 3, 2), dtype=np.float32)
        actions = np.zeros((2, 3, 2, 1), dtype=np.float32)
        latents = np.zeros((2, 3, 2, 2), dtype=np.float32)

        self.sampled_tube.validate_cellwise_rollout_artifact(
            trajectories,
            actions,
            num_cells=2,
            samples_per_cell=3,
            horizon=2,
            state_dim=2,
            latents=latents,
            latent_dim=2,
            z_range=0.8,
            initial_states=trajectories[:, :, 0, :],
        )

        invalid_actions = actions.copy()
        invalid_actions[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "actions.*finite"):
            self.sampled_tube.validate_cellwise_rollout_artifact(
                trajectories,
                invalid_actions,
                num_cells=2,
                samples_per_cell=3,
                horizon=2,
                state_dim=2,
                latents=latents,
                latent_dim=2,
                z_range=0.8,
                initial_states=trajectories[:, :, 0, :],
            )

        invalid_latents = latents.copy()
        invalid_latents[0, 0, 0, 0] = 0.81
        with self.assertRaisesRegex(ValueError, "latent.*range"):
            self.sampled_tube.validate_cellwise_rollout_artifact(
                trajectories,
                actions,
                num_cells=2,
                samples_per_cell=3,
                horizon=2,
                state_dim=2,
                latents=invalid_latents,
                latent_dim=2,
                z_range=0.8,
                initial_states=trajectories[:, :, 0, :],
            )

        different_initial_states = trajectories[:, :, 0, :].copy()
        different_initial_states[0, 0, 0] = np.float32(0.1)
        with self.assertRaisesRegex(ValueError, "initial states"):
            self.sampled_tube.validate_cellwise_rollout_artifact(
                trajectories,
                actions,
                num_cells=2,
                samples_per_cell=3,
                horizon=2,
                state_dim=2,
                latents=latents,
                latent_dim=2,
                z_range=0.8,
                initial_states=different_initial_states,
            )


class ProductionWrapperConfigTests(unittest.TestCase):
    def test_all_environment_model_wrappers_share_states_and_validate(self):
        sampled_tube = importlib.import_module("sampled_tube")
        environments = {
            "cartpole": 3600,
            "mountain_car": 6400,
            "pendulum": 5000,
            "brake_system": 1600,
        }

        for environment, expected_cells in environments.items():
            wrappers = []
            for suffix in ("", "_g_mlp"):
                path = (
                    PROJECT_ROOT
                    / "config"
                    / "sampled_tube"
                    / f"{environment}{suffix}.json"
                )
                self.assertTrue(path.exists(), f"missing wrapper: {path}")
                wrapper = json.loads(path.read_text())
                base = json.loads(
                    (PROJECT_ROOT / wrapper["base_sampling_config"]).read_text()
                )
                model_starv = json.loads(
                    (PROJECT_ROOT / base["starv_config"]).read_text()
                )
                canonical = json.loads(
                    (PROJECT_ROOT / wrapper["grid_config"]).read_text()
                )
                try:
                    metadata = sampled_tube.validate_cellwise_setup(
                        wrapper, base, model_starv, canonical
                    )
                except ValueError as error:
                    self.fail(f"{path} should validate: {error}")
                self.assertEqual(wrapper["samples_per_cell"], 3)
                # The seed value itself is free -- it only picks which
                # samples_per_cell points each cell draws. What must hold is
                # that every wrapper of one environment shares it, so the DWM
                # and cGAN tubes are built from identical initial states
                # (checked after the loop). Pinning a literal here would fail
                # whenever the sweep is rerun under a different seed.
                self.assertIsInstance(wrapper["seed"], int)
                self.assertEqual(
                    len(sampled_tube.grid_cell_bounds(canonical["grid"])),
                    expected_cells,
                )
                self.assertEqual(metadata["model_id"], wrapper["model_id"])
                wrappers.append(wrapper)

            self.assertEqual(wrappers[0]["states_file"], wrappers[1]["states_file"])
            self.assertEqual(wrappers[0]["grid_config"], wrappers[1]["grid_config"])
            # Same states_file only means the same path; the seed must match
            # too, otherwise regenerating that file under one wrapper would
            # silently invalidate the other.
            self.assertEqual(wrappers[0]["seed"], wrappers[1]["seed"])
            self.assertNotEqual(
                wrappers[0]["trajectory_file"], wrappers[1]["trajectory_file"]
            )
            self.assertNotEqual(wrappers[0]["tube_file"], wrappers[1]["tube_file"])


if __name__ == "__main__":
    unittest.main()
