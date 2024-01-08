from tests import _PATH_DATA
from mlops.train_model import load_dataset
import os
import pytest

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    train_ldr, test_ldr = load_dataset(32, path=_PATH_DATA)
    assert len(train_ldr) == 938, "Dataset did not have the correct number of samples"
    assert len(test_ldr) == 157
    assert next(iter(train_ldr))[0].shape == (32, 28, 28)

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("test_input,expected", [(1,1), (16,16), (32,32), (64,64)])
def test_batch_size(test_input, expected):
    train_ldr, _ = load_dataset(test_input, path=_PATH_DATA)
    assert next(iter(train_ldr))[0].shape == (expected, 28, 28), "Batch size was not correct"