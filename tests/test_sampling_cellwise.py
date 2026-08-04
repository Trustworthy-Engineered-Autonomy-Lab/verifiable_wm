import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import sampling


class SamplingEntryPointTests(unittest.TestCase):
    def test_legacy_state_split_generator_is_removed(self):
        self.assertFalse(hasattr(sampling, "generate_dataset"))
        self.assertFalse(hasattr(sampling, "run_sampling_config"))

    def test_main_routes_each_config_directly_to_sampled_tube_generator(self):
        config_path = Path("config/sampling/cartpole.json")
        config = {"model_id": "saliency"}
        with mock.patch.object(
            sampling,
            "parse_args",
            return_value=mock.Mock(configs=[config_path]),
        ), mock.patch.object(
            sampling,
            "load_config",
            return_value=config,
        ), mock.patch.object(
            sampling,
            "generate_sampled_tube",
            return_value=Path("sampled.json"),
        ) as generate:
            sampling.main()

        generate.assert_called_once_with(config, config_path)


class ExplicitLatentRolloutTests(unittest.TestCase):
    def test_rollout_uses_the_explicit_latent_at_each_step(self):
        self.assertIn(
            "latents",
            inspect.signature(sampling.rollout_sampled_trajectories).parameters,
            "rollout_sampled_trajectories must accept explicit latents",
        )

        class Decoder(torch.nn.Module):
            def forward(self, states, z=None):
                values = states[:, :1] if z is None else z[:, :1]
                return values.reshape(-1, 1, 1, 1)

        class Controller(torch.nn.Module):
            def forward(self, images):
                return images.reshape(images.shape[0], -1)[:, :1]

        class Dynamic:
            @staticmethod
            def step(states, actions):
                return states + actions.repeat(1, states.shape[1])

        states0 = torch.zeros(2, 2)
        latents = torch.tensor(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        )
        trajectories, actions = sampling.rollout_sampled_trajectories(
            states0,
            steps=2,
            decoder=Decoder(),
            controller=Controller(),
            dynamic=Dynamic(),
            device=torch.device("cpu"),
            latents=latents,
        )

        torch.testing.assert_close(actions[:, :, 0], latents[:, :, 0])
        torch.testing.assert_close(
            trajectories[:, :, 0],
            torch.tensor([[0.0, 1.0, 3.0], [0.0, 3.0, 7.0]]),
        )

    def test_cell_batch_size_does_not_change_explicit_latent_rollout(self):
        self.assertTrue(
            hasattr(sampling, "rollout_cellwise_batches"),
            "rollout_cellwise_batches is required",
        )
        self.assertIn(
            "progress",
            inspect.signature(sampling.rollout_cellwise_batches).parameters,
            "rollout_cellwise_batches must expose progress updates",
        )

        class Decoder(torch.nn.Module):
            def forward(self, states, z=None):
                values = states[:, :1] if z is None else z[:, :1]
                return values.reshape(-1, 1, 1, 1)

        class Controller(torch.nn.Module):
            def forward(self, images):
                return images.reshape(images.shape[0], -1)[:, :1]

        class Dynamic:
            @staticmethod
            def step(states, actions):
                return states + actions.repeat(1, states.shape[1])

        states = np.zeros((2, 3, 2), dtype=np.float32)
        latents = np.arange(12, dtype=np.float32).reshape(2, 3, 2, 1) / 10.0
        kwargs = {
            "states": states,
            "steps": 2,
            "decoder": Decoder(),
            "controller": Controller(),
            "dynamic": Dynamic(),
            "device": torch.device("cpu"),
            "latents": latents,
        }

        updates = []
        trajectories_1, actions_1 = sampling.rollout_cellwise_batches(
            cell_batch_size=1,
            progress=lambda completed, total: updates.append(
                (completed, total)
            ),
            **kwargs,
        )
        trajectories_2, actions_2 = sampling.rollout_cellwise_batches(
            cell_batch_size=2, **kwargs
        )

        self.assertEqual(trajectories_1.shape, (2, 3, 3, 2))
        self.assertEqual(actions_1.shape, (2, 3, 2, 1))
        np.testing.assert_allclose(trajectories_1, trajectories_2)
        np.testing.assert_allclose(actions_1, actions_2)
        self.assertEqual(updates, [(1, 2), (2, 2)])


class CellwisePipelineSmokeTests(unittest.TestCase):
    def test_invalid_rollout_is_not_published_as_a_trajectory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tube_fields = {
                "grid_config": "grid.json",
                "model_id": "old",
                "samples_per_cell": 3,
                "seed": 2025,
                "cell_batch_size": 1,
                "states_file": str(root / "states.npz"),
                "trajectory_file": str(root / "trajectories.npz"),
                "tube_file": str(root / "tube.json"),
            }
            base = {
                "rollout_steps": 1,
                "device": "cpu",
                "starv_config": "starv.json",
                "decoder": {"name": "Decoder", "weights": "decoder.pth"},
                "controller": {
                    "name": "Controller",
                    "weights": "controller.pth",
                },
                "dynamic": {"name": "Brake"},
            }
            config = {**base, **tube_fields}
            verifier = {
                "name": "BrakeVerifier",
                "kwargs": {"num_steps": 1, "early_stop": False},
            }
            grid = {
                "dims": [
                    {"name": "d", "start": 1.0, "stop": 2.0, "num": 1},
                    {"name": "v", "start": 1.0, "stop": 2.0, "num": 1},
                ]
            }
            starv = {
                "layers": {},
                "verifier": verifier,
                "grid": grid,
            }
            metadata = {
                "model_id": "old",
                "model_type": "Decoder",
                "state_dim": 2,
                "decoder_state_indices": [0, 1],
                "decoder_weights": "decoder.pth",
                "decoder_output_activation": "clamp",
                "controller_name": "Controller",
                "controller_weights": "controller.pth",
                "controller_activation": "sigmoid",
                "dynamic_name": "Brake",
                "dynamic_args": {},
                "formal_semantics_match": True,
                "grid_fingerprint": "fingerprint",
                "horizon": 1,
            }
            bounds = np.array([[[1.0, 2.0], [1.0, 2.0]]])
            states = np.ones((1, 3, 2), dtype=np.float32)
            trajectories = np.ones((1, 3, 2, 2), dtype=np.float32)
            trajectories[0, 0, 1, 0] = np.nan
            actions = np.zeros((1, 3, 1, 1), dtype=np.float32)

            with mock.patch.object(
                sampling,
                "load_config",
                side_effect=[starv, starv],
            ), mock.patch.object(
                sampling,
                "validate_cellwise_setup",
                return_value=metadata,
            ), mock.patch.object(
                sampling,
                "load_controller",
                return_value=object(),
            ), mock.patch.object(
                sampling,
                "load_decoder",
                return_value=object(),
            ), mock.patch.object(
                sampling,
                "grid_cell_bounds",
                return_value=bounds,
            ), mock.patch.object(
                sampling,
                "load_or_create_cellwise_states",
                return_value=states,
            ), mock.patch.object(
                sampling,
                "rollout_cellwise_batches",
                return_value=(trajectories, actions),
            ), mock.patch.object(
                sampling,
                "write_npz_atomic",
            ) as writer:
                with self.assertRaisesRegex(ValueError, "finite"):
                    sampling.generate_sampled_tube(
                        config,
                        root / "sampling.json",
                    )

            writer.assert_not_called()
            self.assertFalse(Path(config["trajectory_file"]).exists())

    def test_dwm_and_g_mlp_generate_compare_compatible_artifacts(self):
        self.assertTrue(
            hasattr(sampling, "generate_sampled_tube"),
            "generate_sampled_tube is required",
        )

        class FakeDecoder(torch.nn.Module):
            def forward(self, states, z=None):
                values = states[:, :1] if z is None else z[:, :1]
                return values.reshape(-1, 1, 1, 1)

        class FakeController(torch.nn.Module):
            def forward(self, images):
                return images.reshape(images.shape[0], -1)[:, :1]

        class FakeDynamic:
            def __init__(self, **kwargs):
                pass

            @staticmethod
            def step(states, actions):
                next_states = states.clone()
                next_states[:, 0] = next_states[:, 0] + actions[:, 0]
                return next_states

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_path = root / "canonical.json"
            dwm_starv_path = root / "dwm_starv.json"
            g_mlp_starv_path = root / "g_mlp_starv.json"
            dwm_base_path = root / "dwm_base.json"
            g_mlp_base_path = root / "g_mlp_base.json"
            states_path = root / "cellwise_states.npz"

            verifier = {
                "name": "MountainCarVerifier",
                "args": [],
                "kwargs": {
                    "goal_position_threshold": -10.0,
                    "num_steps": 2,
                    "early_stop": False,
                },
            }
            grid = {
                "dims": [
                    {"name": "pos", "start": 0.1, "stop": 0.3, "num": 2},
                    {"name": "vel", "start": 0.0, "stop": 0.0, "num": 1},
                ]
            }
            controller_layer = {
                "args": [],
                "kwargs": {
                    "weights": "controller.pth",
                    "activation": "tanh",
                },
            }
            canonical = {
                "layers": {
                    "Decoder": {
                        "args": [],
                        "kwargs": {"weights": "decoder.pth"},
                    },
                    "Controller": controller_layer,
                },
                "verifier": verifier,
                "grid": grid,
                "output_prefix": "unused",
            }
            dwm_starv = canonical
            g_mlp_starv = {
                **canonical,
                "layers": {
                    "G_MLP": {
                        "args": [],
                        "kwargs": {
                            "weights": "g_mlp.pth",
                            "z_range": 0.8,
                        },
                    },
                    "Controller": controller_layer,
                },
            }
            canonical_path.write_text(json.dumps(canonical))
            dwm_starv_path.write_text(json.dumps(dwm_starv))
            g_mlp_starv_path.write_text(json.dumps(g_mlp_starv))

            common_base = {
                "rollout_steps": 2,
                "device": "cpu",
                "controller": {
                    "name": "Controller",
                    "weights": "controller.pth",
                    "args": {"activation": "tanh"},
                },
                "dynamic": {"name": "MountainCar"},
            }
            dwm_base = {
                **common_base,
                "starv_config": str(dwm_starv_path),
                "decoder": {
                    "name": "Decoder",
                    "variant": "old",
                    "weights": "decoder.pth",
                },
            }
            g_mlp_base = {
                **common_base,
                "starv_config": str(g_mlp_starv_path),
                "decoder": {
                    "name": "G_MLP",
                    "variant": "g_mlp",
                    "weights": {"g_mlp": "g_mlp.pth"},
                    "args": {"latent_dim": 2, "z_range": 0.8},
                },
            }
            dwm_base_path.write_text(json.dumps(dwm_base))
            g_mlp_base_path.write_text(json.dumps(g_mlp_base))

            configs = []
            for model_id, base in (
                ("old", dwm_base),
                ("g_mlp", g_mlp_base),
            ):
                configs.append(
                    {
                        **base,
                        "grid_config": str(canonical_path),
                        "model_id": model_id,
                        "samples_per_cell": 3,
                        "seed": 2025,
                        "cell_batch_size": 1,
                        "states_file": str(states_path),
                        "trajectory_file": str(
                            root / f"sampled_trajectories_{model_id}.npz"
                        ),
                        "tube_file": str(root / f"sampled_tube_{model_id}.json"),
                    }
                )

            with mock.patch.dict(
                sampling.__dict__, {"MountainCar": FakeDynamic}
            ), mock.patch.object(
                sampling, "load_decoder", side_effect=lambda config, device: FakeDecoder()
            ), mock.patch.object(
                sampling,
                "load_controller",
                side_effect=lambda config, device: FakeController(),
            ):
                output_paths = [
                    sampling.generate_sampled_tube(
                        config, root / f"{config['model_id']}.json"
                    )
                    for config in configs
                ]

            self.assertEqual(
                output_paths,
                [Path(item["tube_file"]) for item in configs],
            )
            with np.load(states_path, allow_pickle=False) as shared:
                shared_states = np.asarray(shared["states"])
                self.assertEqual(shared_states.shape, (2, 3, 2))

            for config in configs:
                payload = json.loads(Path(config["tube_file"]).read_text())
                self.assertEqual(
                    payload["method"],
                    "cellwise_sampled_trajectory_envelope",
                )
                self.assertEqual(len(payload["cells"]), 2)
                self.assertTrue(
                    all(len(cell["bounds"]) == 3 for cell in payload["cells"])
                )
                expected_layer = (
                    "G_MLP"
                    if config["model_id"] == "g_mlp"
                    else "Decoder"
                )
                self.assertIn(expected_layer, payload["layers"])
                with np.load(
                    config["trajectory_file"], allow_pickle=False
                ) as artifact:
                    self.assertEqual(
                        artifact["trajectories"].shape, (2, 3, 3, 2)
                    )
                    self.assertEqual(artifact["actions"].shape, (2, 3, 2, 1))
                    self.assertNotIn("test_traj", artifact.files)
                    for field in (
                        "state_dim",
                        "decoder_state_indices",
                        "controller_name",
                        "controller_activation",
                        "dynamic_name",
                        "dynamic_args_json",
                        "formal_semantics_match",
                        "source_sampling_config",
                    ):
                        self.assertIn(field, artifact.files)
                    if config["model_id"] == "g_mlp":
                        self.assertEqual(
                            artifact["latents"].shape, (2, 3, 2, 2)
                        )
                        self.assertIn("latent_dim", artifact.files)
                        self.assertIn(
                            "generator_output_activation", artifact.files
                        )
                    else:
                        self.assertIn(
                            "decoder_output_activation", artifact.files
                        )


if __name__ == "__main__":
    unittest.main()
