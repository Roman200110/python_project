import pytest
from src.model import AnimalClassifier

def test_model_initialization():
    num_classes = 10
    model = AnimalClassifier(num_classes).get_model()
    assert model is not None
    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes
