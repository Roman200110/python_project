import pytest
from src.data_loader import CustomDataset, get_transforms, create_dataloaders

def test_custom_dataset():
    dataset_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/data/raw/TAWIRI 02.v3i.coco"  # Update this to your dataset path
    dataset = CustomDataset(root=dataset_path, split='train', transforms=get_transforms(True))
    assert len(dataset) > 0

def test_create_dataloaders():
    dataset_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/data/raw/TAWIRI 02.v3i.coco"  # Update this to your dataset path
    train_loader, valid_loader = create_dataloaders(dataset_path)
    assert len(train_loader) > 0
    assert len(valid_loader) > 0
