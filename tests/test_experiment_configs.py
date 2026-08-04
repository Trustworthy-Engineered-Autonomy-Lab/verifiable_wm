import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTests(unittest.TestCase):
    ENVS = ("cartpole", "mountain_car", "pendulum")

    # The verification grid each environment's paper results are reported on.
    GRIDS = {
        "cartpole": [
            ("pos", 0.0, 0.6, 60),
            ("vel", 0.0, 0.0, 1),
            ("angle", 0.06, 0.12, 60),
            ("avel", 0.0, 0.0, 1),
        ],
        "mountain_car": [
            ("pos", -0.2, 0.6, 80),
            ("vel", 0.0, 0.08, 80),
        ],
        "pendulum": [
            ("theta", 1.0, 2.0, 100),
            ("omega", 4.5, 5.0, 50),
        ],
    }

    # The state box the decoder training images are drawn from. It has to
    # cover the verification grid, otherwise the decoder is extrapolating on
    # exactly the cells being verified.
    DECODER_STATE_SPACES = {
        "cartpole": [(-2.4, 2.4), (-1.0, 1.0), (-0.2095, 0.2095), (-1.0, 1.0)],
        "mountain_car": [(-1.2, 0.6), (-0.08, 0.08)],
        "pendulum": [(-math.pi, math.pi), (-8.0, 8.0)],
    }

    @staticmethod
    def _load(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    def assertFloatClose(self, actual, expected):
        # Tolerance instead of exact equality: importing the StarV stack
        # (pybdr -> codac) changes the process FPU rounding mode, which
        # perturbs all later JSON float parsing by 1 ulp.
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{actual!r} != {expected!r}",
        )

    def test_active_configs_use_t20(self):
        for env in self.ENVS:
            with self.subTest(env=env):
                decoder = self._load(f"config/make_decoder_dataset/{env}.json")
                sampling = self._load(f"config/sampling/{env}.json")
                full = self._load(f"config/starv_verification/{env}.json")

                self.assertEqual(decoder["rollout_steps"], 20)
                self.assertEqual(sampling["rollout_steps"], 20)
                self.assertEqual(full["verifier"]["kwargs"]["num_steps"], 20)

    def test_full_configs_use_approved_grids(self):
        for env in self.ENVS:
            with self.subTest(env=env):
                actual_dims = self._load(
                    f"config/starv_verification/{env}.json"
                )["grid"]["dims"]
                expected_dims = self.GRIDS[env]
                self.assertEqual(len(actual_dims), len(expected_dims))
                for actual, expected in zip(actual_dims, expected_dims):
                    name, start, stop, num = expected
                    self.assertEqual(actual["name"], name)
                    self.assertFloatClose(actual["start"], start)
                    self.assertFloatClose(actual["stop"], stop)
                    self.assertEqual(actual["num"], num)

    def test_decoder_state_spaces_match_contract(self):
        for env in self.ENVS:
            with self.subTest(env=env):
                actual_dims = self._load(
                    f"config/make_decoder_dataset/{env}.json"
                )["state_space"]
                expected_dims = self.DECODER_STATE_SPACES[env]
                self.assertEqual(len(actual_dims), len(expected_dims))
                for actual, (low, high) in zip(actual_dims, expected_dims):
                    self.assertFloatClose(actual["low"], low)
                    self.assertFloatClose(actual["high"], high)

    def test_decoder_state_spaces_cover_the_verification_grid(self):
        for env in self.ENVS:
            with self.subTest(env=env):
                make_config = self._load(f"config/make_decoder_dataset/{env}.json")
                sampling_config = self._load(f"config/sampling/{env}.json")
                starv_config = self._load(sampling_config["starv_config"])

                for training_dim, grid_dim in zip(
                    make_config["state_space"], starv_config["grid"]["dims"]
                ):
                    self.assertLessEqual(training_dim["low"], grid_dim["start"])
                    self.assertGreaterEqual(training_dim["high"], grid_dim["stop"])

    def test_mountain_car_goal_matches_the_grid_stop(self):
        full = self._load("config/starv_verification/mountain_car.json")
        self.assertFloatClose(
            full["verifier"]["kwargs"]["goal_position_threshold"], 0.6
        )


if __name__ == "__main__":
    unittest.main()
