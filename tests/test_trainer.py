import pytest
import torch
from src.model import AnimalClassifier
from src.trainer import Trainer

def test_trainer_initialization():
    num_classes = 10
    model = AnimalClassifier(num_classes).get_model()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)
    device = torch.device("cpu")
    trainer = Trainer(model, optimizer, device)
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.device == device
