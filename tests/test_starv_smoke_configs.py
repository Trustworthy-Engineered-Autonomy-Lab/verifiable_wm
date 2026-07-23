import json
import math
import unittest
from pathlib import Path


class StarVSmokeConfigTests(unittest.TestCase):
    CASES = {
        "cartpole": {
            "checkpoint": (
                "dwm_weight/now_weight/cartpole/alpha_lambda_grid/"
                "alpha_8/lambda_0.1/seed_2025/decoder_best_total.pth"
            ),
            "nums": [4, 1, 4, 1],
            "ranges": [
                (0.18, 0.42),
                (0.0, 0.0),
                (0.078, 0.102),
                (0.0, 0.0),
            ],
        },
        "pendulum": {
            "checkpoint": (
                "dwm_weight/now_weight/pendulum/alpha_lambda_grid/"
                "alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth"
            ),
            "nums": [4, 4],
            "ranges": [(1.03, 1.07), (4.53, 4.57)],
        },
        "mountain_car": {
            "checkpoint": (
                "dwm_weight/now_weight/mountain_car/alpha_lambda_grid/"
                "background/alpha_16/lambda_0.5/seed_2025/"
                "decoder_best_total.pth"
            ),
            "nums": [4, 4],
            "ranges": [(0.03, 0.07), (0.003, 0.007)],
        },
    }

    def test_smoke_configs_use_selected_checkpoints_and_sixteen_isolated_cells(
        self,
    ):
        for env, expected in self.CASES.items():
            with self.subTest(env=env):
                path = Path("config/starv_verification/smoke") / f"{env}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                dims = config["grid"]["dims"]

                self.assertEqual(
                    config["layers"]["Decoder"]["kwargs"]["weights"],
                    expected["checkpoint"],
                )
                self.assertEqual([dim["num"] for dim in dims], expected["nums"])
                self.assertEqual(math.prod(expected["nums"]), 16)
                self.assertEqual(
                    [(dim["start"], dim["stop"]) for dim in dims],
                    expected["ranges"],
                )
                self.assertEqual(config["verifier"]["kwargs"]["num_steps"], 30)
                self.assertTrue(config["verifier"]["kwargs"]["save_history"])
                self.assertFalse(config["verifier"]["kwargs"]["early_stop"])
                self.assertEqual(
                    config["output_prefix"],
                    f"results/smoke/{env}/safety_result",
                )


if __name__ == "__main__":
    unittest.main()
