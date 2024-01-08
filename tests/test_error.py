import pytest
import torch

from mlops.models.model import NeuralNet


def test_model():
    model = NeuralNet()
    model.eval()
    x = torch.rand(1, 1, 28, 28)
    with pytest.raises(ValueError, match="Expected input to a 3D tensor"):
        model(x)
