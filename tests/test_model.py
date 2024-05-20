import pytest
import torch
from src.model import AnimalClassifier

@pytest.fixture
def model():
    return AnimalClassifier(num_classes=4)

def test_model_forward(model):
    inputs = torch.randn(1, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (1, 4)
