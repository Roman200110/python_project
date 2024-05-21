import pytest
from src.data_loader import CustomDataset, get_transforms, create_dataloaders

def test_custom_dataset():
    dataset_path = "path/to/your/dataset"  # Update this to your dataset path
    dataset = CustomDataset(root=dataset_path, split='train', transforms=get_transforms(True))
    assert len(dataset) > 0

def test_create_dataloaders():
    dataset_path = "path/to/your/dataset"  # Update this to your dataset path
    train_loader, valid_loader, test_loader = create_dataloaders(dataset_path)
    assert len(train_loader) > 0
    assert len(valid_loader) > 0
    assert len(test_loader) > 0
