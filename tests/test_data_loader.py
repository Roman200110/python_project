import pytest
from src.data_loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader("data/raw/train_val2019")

def test_load_data(data_loader):
    train_loader, val_loader = data_loader.load_data(batch_size=32)
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
