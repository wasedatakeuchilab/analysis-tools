import os
import shutil
import tempfile
import unittest

import papermill

from tests import fake


class TestNotebook(unittest.TestCase):
    filepath: str = os.path.abspath("notebooks/carrier_relaxation.ipynb")

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
        self._test_filepath = os.path.join(self._test_dir, "dummy_data.img")
        data = fake.make_trpl_data()
        with open(self._test_filepath, "wb") as f:
            f.write(data.to_raw_binary())

    def tearDown(self) -> None:
        os.chdir(self._curdir)
        shutil.rmtree(self._test_dir)

    def test_notebook_is_executable(self) -> None:
        try:
            papermill.execute_notebook(
                self.filepath,
                None,
                parameters={"file": self._test_filepath},
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
                    "file": self._test_filepath,
                    "outputdir": outputdir,
                },
                progress_bar=False,
            )
        except Exception as err:
            self.fail(err)
        self.assertTrue(os.path.exists(outputdir))
        output_files = os.listdir(outputdir)
        self.assertEqual(len(output_files), 2)
        for file in output_files:
            self.assertTrue(file.endswith(".csv"))

    def test_with_wavelength_range(self) -> None:
        wavelength_range = [920, 1000]
        try:
            papermill.execute_notebook(
                self.filepath,
                None,
                parameters={
                    "file": self._test_filepath,
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
                    "file": self._test_filepath,
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
