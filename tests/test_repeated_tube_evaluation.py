import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import repeated_tube_evaluation as repeated


class AggregateTests(unittest.TestCase):
    def test_aggregate_values_uses_population_variance_and_standard_deviation(self):
        result = repeated.aggregate_values([0.5, 1.0])

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["ddof"], 0)
        self.assertEqual(result["mean"], 0.75)
        self.assertEqual(result["variance"], 0.0625)
        self.assertEqual(result["std"], 0.25)

    def test_duplicate_names_inside_one_group_are_rejected(self):
        repeats = [
            {
                "name": "repeat_0",
                "group": "raw",
                "inflate": False,
                "tube": "tube.json",
                "evaluation_real": "real.npz",
            },
            {
                "name": "repeat_0",
                "group": "raw",
                "inflate": False,
                "tube": "tube.json",
                "evaluation_real": "real.npz",
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate repeat"):
            repeated.validate_repeats(repeats)


class RepeatedConfigTests(unittest.TestCase):
    def write_config(self, root: Path, repeats: list[dict]) -> Path:
        path = root / "postprocess.json"
        path.write_text(json.dumps({
            "tube": "safety/default_tube.json",
            "real_trajectories": "safety/default_real.npz",
            "check_dims": [0, 2],
            "evaluation": {
                "real_key": "test_traj",
                "output_dir": "results/evaluation",
            },
            "inflation": {
                "calibration_key": "val_traj",
                "alpha": 0.05,
                "output": "results/inflated.json",
                "calibration_output": "results/calibration.json",
            },
            "repeated_evaluation": {
                "output_dir": "results/repeated",
                "repeats": repeats,
            },
        }), encoding="utf-8")
        return path

    def test_load_repeated_config_inherits_shared_postprocess_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root, [
                {"name": "repeat_0", "group": "raw", "inflate": False},
                {"name": "repeat_0", "group": "inflated", "inflate": True},
            ])

            config = repeated.load_repeated_config(path, project_root=root)

        self.assertEqual(config["dims"], (0, 2))
        self.assertEqual(config["output_dir"], root / "results/repeated")
        self.assertEqual(config["repeats"][0], {
            "name": "repeat_0",
            "group": "raw",
            "inflate": False,
            "tube": root / "safety/default_tube.json",
            "evaluation_real": root / "safety/default_real.npz",
            "evaluation_key": "test_traj",
        })
        self.assertEqual(config["repeats"][1], {
            "name": "repeat_0",
            "group": "inflated",
            "inflate": True,
            "tube": root / "safety/default_tube.json",
            "evaluation_real": root / "safety/default_real.npz",
            "evaluation_key": "test_traj",
            "calibration_real": root / "safety/default_real.npz",
            "calibration_key": "val_traj",
            "alpha": 0.05,
        })

    def test_load_repeated_config_applies_per_repeat_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root, [{
                "name": "repeat_1",
                "group": "inflated",
                "inflate": True,
                "tube": "other/tube.json",
                "evaluation_real": "other/test.npz",
                "evaluation_key": "held_out",
                "calibration_real": "other/val.npz",
                "calibration_key": "calibration",
                "alpha": 0.1,
            }])

            config = repeated.load_repeated_config(path, project_root=root)

        self.assertEqual(config["repeats"], [{
            "name": "repeat_1",
            "group": "inflated",
            "inflate": True,
            "tube": root / "other/tube.json",
            "evaluation_real": root / "other/test.npz",
            "evaluation_key": "held_out",
            "calibration_real": root / "other/val.npz",
            "calibration_key": "calibration",
            "alpha": 0.1,
        }])

    def test_main_loads_repeats_from_positional_config(self):
        loaded = {
            "repeats": [{"name": "repeat_0"}],
            "dims": (0, 1),
            "output_dir": Path("results"),
        }
        payload = {"groups": {}}
        with mock.patch.object(
            repeated, "load_repeated_config", return_value=loaded
        ) as load_config, mock.patch.object(
            repeated, "run_repeated_evaluation", return_value=payload
        ) as run:
            repeated.main(["config.json", "--overwrite"])

        load_config.assert_called_once_with(Path("config.json"))
        run.assert_called_once_with(**loaded, overwrite=True)


class RepeatedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tube_path = self.root / "tube.json"
        self.real_path = self.root / "real.npz"
        tube = {
            "layers": {"Decoder": {"kwargs": {"weights": "decoder.pth"}}},
            "verifier": {"name": "ToyVerifier", "kwargs": {}},
            "grid": {"dims": [
                {"name": "x", "start": 0.0, "stop": 1.0, "num": 1},
                {"name": "y", "start": 0.0, "stop": 1.0, "num": 1},
            ]},
            "cells": [{"bounds": [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]}],
        }
        self.tube_path.write_text(json.dumps(tube), encoding="utf-8")

        inside = np.full((10, 2, 2), 0.5)
        half = inside.copy()
        half[:5, 1, 0] = 1.2
        calibration = np.full((20, 2, 2), 0.5)
        calibration[:, 1, 0] = 1.2
        inflated_test = np.full((10, 2, 2), 0.5)
        inflated_test[:, 1, 0] = 1.15
        np.savez_compressed(
            self.real_path,
            inside=inside,
            half=half,
            calibration=calibration,
            inflated_test=inflated_test,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_repeats(self):
        return [
            {
                "name": "repeat_0",
                "group": "raw",
                "inflate": False,
                "tube": self.tube_path,
                "evaluation_real": self.real_path,
                "evaluation_key": "inside",
            },
            {
                "name": "repeat_1",
                "group": "raw",
                "inflate": False,
                "tube": self.tube_path,
                "evaluation_real": self.real_path,
                "evaluation_key": "half",
            },
            {
                "name": "repeat_0",
                "group": "inflated",
                "inflate": True,
                "tube": self.tube_path,
                "calibration_real": self.real_path,
                "calibration_key": "calibration",
                "evaluation_real": self.real_path,
                "evaluation_key": "inflated_test",
                "alpha": 0.1,
            },
        ]

    def test_arbitrary_raw_and_inflated_repeats_are_grouped_and_aggregated(self):
        output_dir = self.root / "results"

        payload = repeated.run_repeated_evaluation(
            repeats=self.make_repeats(),
            dims=(0, 1),
            output_dir=output_dir,
            overwrite=False,
        )

        self.assertEqual(len(payload["repeats"]), 3)
        raw = payload["groups"]["raw"]
        self.assertEqual(raw["num_repeats"], 2)
        self.assertEqual(raw["coverage"], {
            "mean": 0.75,
            "variance": 0.0625,
            "std": 0.25,
            "count": 2,
            "ddof": 0,
        })
        self.assertEqual(raw["area"]["mean"], 1.0)
        self.assertEqual(raw["area"]["variance"], 0.0)
        inflated = payload["groups"]["inflated"]
        self.assertEqual(inflated["coverage"]["mean"], 1.0)
        self.assertAlmostEqual(inflated["area"]["mean"], 1.96)

        repeat_dir = output_dir / "inflated" / "repeat_0"
        self.assertTrue((repeat_dir / "inflated_tube.json").is_file())
        self.assertTrue((repeat_dir / "calibration.json").is_file())
        self.assertTrue((repeat_dir / "metrics.json").is_file())
        self.assertFalse(list(output_dir.rglob("*.png")))

    def test_tables_contain_repeat_details_and_display_formatted_aggregates(self):
        output_dir = self.root / "results"
        repeated.run_repeated_evaluation(
            repeats=self.make_repeats(),
            dims=(0, 1),
            output_dir=output_dir,
            overwrite=False,
        )

        with (output_dir / "repeat_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            detail_rows = list(csv.DictReader(file))
        with (output_dir / "aggregate_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            aggregate_rows = {
                row["Group"]: row for row in csv.DictReader(file)
            }

        self.assertEqual(len(detail_rows), 3)
        self.assertEqual(detail_rows[0]["Coverage"], "100.00%")
        self.assertEqual(aggregate_rows["raw"]["Coverage Mean"], "75.00%")
        self.assertEqual(aggregate_rows["raw"]["Coverage Variance (%²)"], "625.000000")
        self.assertEqual(aggregate_rows["raw"]["Coverage Std"], "25.00%")
        self.assertEqual(aggregate_rows["raw"]["Area Mean"], "1")
        self.assertTrue((output_dir / "repeat_results.json").is_file())

    def test_existing_output_is_rejected_before_any_repeat_is_rerun(self):
        output_dir = self.root / "results"
        repeated.run_repeated_evaluation(
            repeats=self.make_repeats(),
            dims=(0, 1),
            output_dir=output_dir,
            overwrite=False,
        )

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            repeated.run_repeated_evaluation(
                repeats=self.make_repeats(),
                dims=(0, 1),
                output_dir=output_dir,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
