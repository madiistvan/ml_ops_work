import torch

from mlops.models.model import NeuralNet


def test_model():
    model = NeuralNet()
    model.eval()
    x = torch.rand(1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
