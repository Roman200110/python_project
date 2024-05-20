import pytest
from src.trainer import Trainer
from src.model import AnimalClassifier
from src.data_loader import DataLoader
import torch

@pytest.fixture
def trainer():
    data_loader = DataLoader("data/raw/train_val2019")
    train_loader, val_loader = data_loader.load_data(batch_size=32)
    model = AnimalClassifier(num_classes=4)
    return Trainer(model, train_loader, val_loader)

def test_train(trainer):
    trainer.train(epochs=1)

def test_evaluate(trainer):
    accuracy = trainer.evaluate()
    assert 0 <= accuracy <= 100
