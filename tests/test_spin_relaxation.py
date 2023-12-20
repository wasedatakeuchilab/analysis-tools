import os
import shutil
import tempfile
import unittest

import papermill

from tests import fake


class TestNotebook(unittest.TestCase):
    filepath: str = os.path.abspath("notebooks/spin_relaxation.ipynb")

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["PYDEVD_DISABLE_FILE_VALIDATION"]

    def setUp(self) -> None:
        self._curdir = os.path.curdir
        self._test_dir = tempfile.mkdtemp()
        os.chdir(self._test_dir)
        self._test_filepath_RR = os.path.join(self._test_dir, "dummy_data_RR.img")
        self._test_filepath_RL = os.path.join(self._test_dir, "dummy_data_RR.img")
        data_RR = fake.make_trpl_data(tau=0.2)
        data_RL = fake.make_trpl_data(tau=0.225)
        data_RL.df["intensity"] *= (
            0.8 * data_RR.intensity.sum() / data_RL.intensity.sum()
        )
        with open(self._test_filepath_RR, "wb") as f:
            f.write(data_RR.to_raw_binary())
        with open(self._test_filepath_RL, "wb") as f:
            f.write(data_RL.to_raw_binary())

    def tearDown(self) -> None:
        os.chdir(self._curdir)
        shutil.rmtree(self._test_dir)

    def test_notebook_is_executable(self) -> None:
        try:
            papermill.execute_notebook(
                self.filepath,
                None,
                parameters={
                    "RR": self._test_filepath_RR,
                    "RL": self._test_filepath_RL,
                },
                progress_bar=False,
            )
        except Exception as err:
            self.fail(err)

    def test_with_outputdir(self) -> None:
        outputdir = "output"
        try:
            papermill.execute_notebook(
                self.filepath,
                None,
                parameters={
                    "RR": self._test_filepath_RR,
                    "RL": self._test_filepath_RL,
                    "outputdir": outputdir,
                },
                progress_bar=False,
            )
        except Exception as err:
            self.fail(err)
        self.assertTrue(os.path.exists(outputdir))
        output_files = os.listdir(outputdir)
        self.assertEqual(len(output_files), 1)
        for file in output_files:
            self.assertTrue(file.endswith(".csv"))

    def test_with_wavelength_range(self) -> None:
        wavelength_range = [920, 1000]
        try:
            papermill.execute_notebook(
                self.filepath,
                None,
                parameters={
                    "RR": self._test_filepath_RR,
                    "RL": self._test_filepath_RL,
                    "wavelength_range": wavelength_range,
                },
                progress_bar=False,
            )
        except Exception as err:
            self.fail(err)

    def test_with_dump_csv_is_false(self) -> None:
        try:
            papermill.execute_notebook(
                self.filepath,
                None,
                parameters={
                    "RR": self._test_filepath_RR,
                    "RL": self._test_filepath_RL,
                    "dump_csv": False,
                },
                progress_bar=False,
            )
        except Exception as err:
            self.fail(err)
        csv_files = list(
            filter(
                lambda file: os.path.isfile(file) and file.endswith(".csv"),
                os.listdir("."),
            )
        )
        self.assertEqual(len(csv_files), 0)
