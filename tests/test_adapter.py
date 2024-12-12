import json
import pickle
import sys
import unittest
from pathlib import Path

import dtlpy as dl

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from adapters.yolo_world.model_adapter import Adapter  # noqa: E402 (ignores module level import not at top of file)

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


class MockItem:
    """Mock class to simulate a dl.Item object for testing purposes."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.file_name_no_ext = Path(file_name).stem
        self.file_path = TEST_DATA_DIR / file_name

    def download(self, overwrite=True):
        """Mock download method returning the local file path."""
        return str(self.file_path)


class TestAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Load the 'dataloop.json' configuration
        with open("dataloop.json", "r") as f:
            config = json.load(f)

        model_json = config.get("components", {}).get("models", [])[0]
        model = dl.Model.from_json(
            _json=model_json,
            client_api=dl.ApiClient(),
            project=None,
            package=dl.Package(),
        )
        cls.adapter = Adapter(model_entity=model)

    def load_expected_output(self, file_name):
        """Helper function to load expected output from a pickle file."""
        expected_output_path = TEST_DATA_DIR / file_name
        with open(expected_output_path, "rb") as f:
            return pickle.load(f)

    def test_load_model(self):
        """Test the adapter's model loading functionality."""
        self.adapter.load(local_path=".")
        expected = self.load_expected_output("test_load_model.pkl")
        self.assertEqual(
            self.adapter.model.names, expected, "Model labels do not match expected labels."
        )

    def test_prepare_item_func(self):
        """Test the adapter's item preparation function."""
        file_names = ["test_1.jpg"]
        for file_name in file_names:
            with self.subTest(file_name=file_name):
                mock_item = MockItem(file_name=file_name)
                image = self.adapter.prepare_item_func(item=mock_item)
                expected = self.load_expected_output(
                    f"test_prepare_item_func_{mock_item.file_name_no_ext}.pkl"
                )
                self.assertEqual(
                    image,
                    expected,
                    f"Prepared image for {mock_item.file_name} does not match expected output.",
                )

    def test_predict(self):
        """Test the adapter's prediction functionality."""
        file_names = ["test_1.jpg"]
        self.adapter.load(local_path=".")
        for file_name in file_names:
            with self.subTest(file_name=file_name):
                mock_item = MockItem(file_name=file_name)
                image = self.adapter.prepare_item_func(item=mock_item)
                results = self.adapter.predict([image])
                expected = self.load_expected_output(
                    f"test_predict_{mock_item.file_name_no_ext}.pkl"
                )
                self.assertEqual(
                    results,
                    expected,
                    f"Prediction results for {mock_item.file_name} do not match expected output.",
                )


if __name__ == "__main__":
    unittest.main()
