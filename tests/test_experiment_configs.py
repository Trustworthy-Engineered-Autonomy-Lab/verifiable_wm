import json
import math
import unittest
from pathlib import Path


class ExperimentConfigTests(unittest.TestCase):
    ENVS = ("cartpole", "mountain_car", "pendulum")
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

    @staticmethod
    def _load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def assertFloatClose(self, actual, expected):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{actual!r} != {expected!r}",
        )

    def test_active_configs_use_t20(self):
        for env in self.ENVS:
            with self.subTest(env=env):
                decoder = self._load(
                    Path("config/make_decoder_dataset") / f"{env}.json"
                )
                sampling = self._load(Path("config/sampling") / f"{env}.json")
                full = self._load(
                    Path("config/starv_verification") / f"{env}.json"
                )
                smoke = self._load(
                    Path("config/starv_verification/smoke") / f"{env}.json"
                )

                self.assertEqual(decoder["rollout_steps"], 20)
                self.assertEqual(sampling["rollout_steps"], 20)
                self.assertEqual(full["verifier"]["kwargs"]["num_steps"], 20)
                self.assertEqual(smoke["verifier"]["kwargs"]["num_steps"], 20)

    def test_full_configs_use_approved_grids(self):
        for env in self.ENVS:
            with self.subTest(env=env):
                full = self._load(
                    Path("config/starv_verification") / f"{env}.json"
                )
                actual_dims = full["grid"]["dims"]
                expected_dims = self.GRIDS[env]
                self.assertEqual(len(actual_dims), len(expected_dims))
                for actual, expected in zip(actual_dims, expected_dims):
                    name, start, stop, num = expected
                    self.assertEqual(actual["name"], name)
                    self.assertFloatClose(actual["start"], start)
                    self.assertFloatClose(actual["stop"], stop)
                    self.assertEqual(actual["num"], num)

    def test_mountain_car_decoder_range_and_goal_match_contract(self):
        decoder = self._load("config/make_decoder_dataset/mountain_car.json")
        full = self._load("config/starv_verification/mountain_car.json")
        state_space = {
            dim["name"]: (dim["low"], dim["high"])
            for dim in decoder["state_space"]
        }

        for actual, expected in zip(state_space["position"], (-1.2, 0.6)):
            self.assertFloatClose(actual, expected)
        for actual, expected in zip(state_space["velocity"], (-0.08, 0.08)):
            self.assertFloatClose(actual, expected)
        self.assertFloatClose(
            full["verifier"]["kwargs"]["goal_position_threshold"], 0.6
        )


if __name__ == "__main__":
    unittest.main()
