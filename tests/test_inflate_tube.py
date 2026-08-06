import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import evaluate_tube as evaluation
import inflate_tube as inflation


class InflateTubeTests(unittest.TestCase):
    def test_conformal_quantile_uses_finite_sample_rank(self):
        quantile, rank = inflation.conformal_quantile_with_rank(
            np.arange(20, dtype=float), alpha=0.1
        )

        self.assertEqual(rank, 19)
        self.assertEqual(quantile, 18.0)

    def test_inflate_cells_preserves_initial_cells_and_expands_future_bounds(self):
        cells = [{"bounds": [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.2, 0.8], [2.2, 2.8]],
        ]}]

        inflated = inflation.inflate_cells(cells, dims=(0, 1), epsilons=(0.1, 0.1))

        self.assertEqual(inflated[0]["bounds"][0], cells[0]["bounds"][0])
        np.testing.assert_allclose(
            inflated[0]["bounds"][1],
            [[0.1, 0.9], [2.1, 2.9]],
        )
        self.assertEqual(cells[0]["bounds"][1], [[0.2, 0.8], [2.2, 2.8]])

    def test_run_inflation_preserves_starv_schema_and_writes_sidecar(self):
        payload = {
            "layers": {"Decoder": {"kwargs": {"weights": "decoder.pth"}}},
            "verifier": {"name": "ToyVerifier", "kwargs": {}},
            "grid": {"dims": [
                {"name": "x", "start": 0.0, "stop": 1.0, "num": 1},
                {"name": "y", "start": 0.0, "stop": 1.0, "num": 1},
            ]},
            "output_prefix": "toy",
            "cells": [{"bounds": [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]}],
        }
        calibration = np.full((20, 2, 2), 0.5)
        calibration[:, 1, 0] = 1.2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tube_path = root / "raw.json"
            real_path = root / "real.npz"
            output_path = root / "inflated.json"
            sidecar_path = root / "calibration.json"
            tube_path.write_text(json.dumps(payload), encoding="utf-8")
            np.savez_compressed(real_path, val_traj=calibration)

            result = inflation.run_inflation(
                tube_path=tube_path,
                calibration_path=real_path,
                calibration_key="val_traj",
                dims=(0, 1),
                alpha=0.1,
                output_path=output_path,
                calibration_output_path=sidecar_path,
                overwrite=False,
            )
            written_tube = json.loads(output_path.read_text(encoding="utf-8"))
            written_calibration = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.assertEqual(set(written_tube), set(payload))
        self.assertEqual(written_tube["grid"], payload["grid"])
        self.assertEqual(written_tube["cells"][0]["bounds"][0], payload["cells"][0]["bounds"][0])
        np.testing.assert_allclose(
            written_tube["cells"][0]["bounds"][1],
            [[-0.2, 1.2], [-0.2, 1.2]],
        )
        self.assertAlmostEqual(result["gamma"], 0.2)
        self.assertEqual(written_calibration["rank"], 19)
        self.assertEqual(written_calibration["check_dims"], [0, 1])

    def test_main_reads_inflation_section_from_positional_config(self):
        loaded = {
            "tube_path": Path("raw.json"),
            "calibration_path": Path("real.npz"),
            "calibration_key": "val_traj",
            "dims": (0, 1),
            "alpha": 0.05,
            "output_path": Path("inflated.json"),
            "calibration_output_path": Path("calibration.json"),
        }
        with mock.patch.object(
            evaluation, "load_postprocess_config", return_value=loaded
        ) as load_config, mock.patch.object(
            inflation, "run_inflation", return_value={
                "gamma": 0.2,
                "inflated_tube": "inflated.json",
            }
        ) as run:
            inflation.main(["config.json", "--overwrite"])

        load_config.assert_called_once_with(Path("config.json"), "inflation")
        run.assert_called_once_with(**loaded, overwrite=True)


if __name__ == "__main__":
    unittest.main()
