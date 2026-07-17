import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from starv_verification.model import FullModel
from starv_verification.verifiers import PendulumVerifier


class StarVLPSolverTests(unittest.TestCase):
    def test_pendulum_smoke_routes_linprog_through_model_and_dynamics(self):
        config = json.loads(
            Path("config/starv_verification/smoke/pendulum.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config["layers"]["Decoder"]["kwargs"]["lp_solver"], "linprog"
        )
        self.assertEqual(
            config["layers"]["Controller"]["kwargs"]["lp_solver"], "linprog"
        )
        self.assertEqual(config["verifier"]["kwargs"]["lp_solver"], "linprog")

        model = FullModel(layers=config["layers"])
        verifier = PendulumVerifier(**config["verifier"]["kwargs"])
        state_bound = np.array([[1.03, 4.53], [1.04, 4.54]], dtype=np.float32)

        with mock.patch(
            "gurobipy.Model",
            side_effect=AssertionError("Pendulum smoke reached Gurobi"),
        ):
            action_bound = model.reach(state_bound)
            next_bound = verifier.dynamic_step(
                np.concatenate([state_bound, action_bound], axis=1)
            )

        self.assertEqual(action_bound.shape, (2, 1))
        self.assertEqual(next_bound.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
