import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "starv_verification.ipynb"


class StarvVerificationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.sources = ["".join(cell["source"]) for cell in cls.notebook["cells"]]

    def test_notebook_is_clean_and_uses_starv_kernel(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(
            self.notebook["metadata"]["kernelspec"]["display_name"],
            "starv_shared",
        )
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")

    def test_three_environments_have_independent_launch_cells(self):
        for env_name in ("cartpole", "mountain_car", "pendulum"):
            expected = f'start_verification("{env_name}")'
            self.assertEqual(sum(expected in source for source in self.sources), 1)
        self.assertFalse(any("start_all" in source for source in self.sources))

    def test_default_launch_uses_128_ranks_and_detaches(self):
        helper_source = next(
            source for source in self.sources if "def start_verification" in source
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config" / "starv_verification" / "cartpole.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}", encoding="utf-8")
            mpirun = root / "mpirun"
            python = root / "python"
            mpirun.touch(mode=0o755)
            python.touch(mode=0o755)

            namespace = {"NOTEBOOK_DIR": root}
            exec(helper_source, namespace)
            namespace["MPIRUN"] = mpirun
            namespace["STARV_PYTHON"] = python

            process = mock.Mock(pid=4321)
            with mock.patch.object(
                namespace["subprocess"], "Popen", return_value=process
            ) as popen:
                with redirect_stdout(io.StringIO()):
                    pid = namespace["start_verification"]("cartpole")

            self.assertEqual(pid, 4321)
            command = popen.call_args.args[0]
            self.assertEqual(
                command,
                [
                    str(mpirun),
                    "-np",
                    "128",
                    str(python),
                    "verify.py",
                    "config/starv_verification/cartpole.json",
                ],
            )
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["cwd"], root)
            self.assertIs(kwargs["stderr"], namespace["subprocess"].STDOUT)
            self.assertIs(kwargs["stdin"], namespace["subprocess"].DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "GRB_THREADS"):
                self.assertEqual(kwargs["env"][name], "1")
            self.assertEqual(
                Path(kwargs["stdout"].name),
                root / "log" / "cartpole_verify.out",
            )
            self.assertTrue(kwargs["stdout"].closed)
            self.assertEqual(namespace["VERIFICATION_PIDS"], {"cartpole": 4321})

    def test_launch_rejects_invalid_inputs_before_popen(self):
        helper_source = next(
            source for source in self.sources if "def start_verification" in source
        )
        namespace = {"NOTEBOOK_DIR": ROOT}
        exec(helper_source, namespace)
        with mock.patch.object(namespace["subprocess"], "Popen") as popen:
            with self.assertRaises(ValueError):
                namespace["start_verification"]("unknown")
            with self.assertRaises(ValueError):
                namespace["start_verification"]("cartpole", nproc=0)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
