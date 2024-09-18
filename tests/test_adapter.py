import json
import pickle
import sys
from pathlib import Path

import dtlpy as dl
import pytest

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from model_adapter import Adapter  # noqa: E402 (ignores module level import not at top of file)

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def adapter():
    """Fixture to initialize the Adapter with a model loaded from 'dataloop.json'."""
    with open("dataloop.json", "r") as f:
        config = json.load(f)

    model_json = config.get("components", {}).get("models", [])[0]
    model = dl.Model.from_json(
        _json=model_json,
        client_api=dl.ApiClient(),
        project=None,
        package=dl.Package(),
    )
    return Adapter(model_entity=model)


@pytest.fixture
def mock_item(request):
    """Fixture to provide a mock item based on the parameterized file name."""

    class MockItem:
        """Mock class to simulate a dl.Item object for testing purposes."""

        def __init__(self, file_name):
            self.file_name = file_name
            self.file_name_no_ext = Path(file_name).stem
            self.file_path = TEST_DATA_DIR / file_name

        def download(self, overwrite=True):
            """Mock download method returning the local file path."""
            return str(self.file_path)

    return MockItem(file_name=request.param)


def load_expected_output(file_name):
    """Helper function to load expected output from a pickle file."""
    expected_output_path = TEST_DATA_DIR / file_name
    with open(expected_output_path, "rb") as f:
        return pickle.load(f)


def test_load_model(adapter):
    """Test the adapter's model loading functionality."""
    adapter.load(local_path=".")
    expected = load_expected_output("test_load_model.pkl")
    assert (
        adapter.model.names == expected
    ), "Model labels do not match expected labels."


@pytest.mark.parametrize("mock_item", ["test_1.jpg"], indirect=True)
def test_prepare_item_func(adapter, mock_item):
    """Test the adapter's item preparation function."""
    image = adapter.prepare_item_func(item=mock_item)
    expected = load_expected_output(
        "test_prepare_item_func_{}.pkl".format(mock_item.file_name_no_ext)
    )
    assert image == expected, "Prepared image for {} does not match expected output.".format(mock_item.file_name)


@pytest.mark.parametrize("mock_item", ["test_1.jpg"], indirect=True)
def test_predict(adapter, mock_item):
    """Test the adapter's prediction functionality."""
    adapter.load(local_path=".")
    image = adapter.prepare_item_func(item=mock_item)
    results = adapter.predict([image])
    expected = load_expected_output("test_predict_{}.pkl".format(mock_item.file_name_no_ext))
    assert results == expected, "Prediction results for {} do not match expected output.".format(mock_item.file_name)
