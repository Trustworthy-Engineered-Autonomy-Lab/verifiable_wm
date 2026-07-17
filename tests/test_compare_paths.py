import argparse
import os
import tempfile
import unittest
from pathlib import Path

import compare


class ComparePathResolutionTests(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "env": "pendulum",
            "safety": None,
            "real": None,
            "dwm": None,
            "outdir": None,
            "real_key": "test_traj",
            "dwm_key": "test_traj",
            "plot_dims": None,
            "check_dims": None,
            "max_steps": None,
            "delta": 0.0,
            "print_keys": False,
            "dpi": 230,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_missing_path_arguments_use_module_defaults(self):
        compare.apply_args(self.make_args())

        self.assertEqual(compare.SAFETY_PATH, compare.DEFAULT_SAFETY_PATH)
        self.assertEqual(compare.REAL_TRAJ_PATH, compare.DEFAULT_REAL_TRAJ_PATH)
        self.assertEqual(compare.DWM_TRAJ_PATH, compare.DEFAULT_DWM_TRAJ_PATH)
        self.assertEqual(compare.OUT_DIR, compare.DEFAULT_OUT_DIR)

    def test_explicit_paths_override_defaults_independently(self):
        safety = Path("custom/safety.json")
        compare.apply_args(self.make_args(safety=safety))

        self.assertEqual(compare.SAFETY_PATH, safety)
        self.assertEqual(compare.REAL_TRAJ_PATH, compare.DEFAULT_REAL_TRAJ_PATH)
        self.assertEqual(compare.DWM_TRAJ_PATH, compare.DEFAULT_DWM_TRAJ_PATH)
        self.assertEqual(compare.OUT_DIR, compare.DEFAULT_OUT_DIR)

    def test_defaults_do_not_depend_on_working_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                compare.apply_args(self.make_args())
            finally:
                os.chdir(original_cwd)

        self.assertEqual(compare.SAFETY_PATH, compare.DEFAULT_SAFETY_PATH)
        self.assertTrue(compare.DEFAULT_SAFETY_PATH.is_absolute())
        self.assertEqual(compare.PROJECT_ROOT, Path(compare.__file__).resolve().parent)

    def test_env_selects_dimensions_without_rebuilding_paths(self):
        compare.apply_args(self.make_args(env="cartpole"))

        self.assertEqual(compare.PLOT_DIMS, (0, 2))
        self.assertEqual(compare.CHECK_DIMS, (0, 2))
        self.assertEqual(compare.SAFETY_PATH, compare.DEFAULT_SAFETY_PATH)
        self.assertEqual(compare.REAL_TRAJ_PATH, compare.DEFAULT_REAL_TRAJ_PATH)


if __name__ == "__main__":
    unittest.main()
