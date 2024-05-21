import pytest
import torch
from src.utils import save_checkpoint, load_checkpoint
from src.model import AnimalClassifier

def test_save_and_load_checkpoint():
    num_classes = 10
    model = AnimalClassifier(num_classes).get_model()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)
    state = {'epoch': 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    save_checkpoint(state, "test_checkpoint.pth.tar")

    loaded_epoch = load_checkpoint("test_checkpoint.pth.tar", model, optimizer)
    assert loaded_epoch == 1
