import torch
from src.model import AnimalClassifier
from src.data_loader import create_dataloaders
from src.trainer import Trainer
from pycocotools.coco import COCO
import os

def train():
    """
    The main function to run the training process.
    """
    dataset_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/data/raw/TAWIRI 02.v3i.coco"  # Update this to your dataset path
    num_classes = len(COCO(os.path.join(dataset_path, "train", "_annotations_bbox.coco.json")).cats.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AnimalClassifier(num_classes).get_model().to(device)

    train_loader, valid_loader = create_dataloaders(dataset_path)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)

    coco_gt = COCO(os.path.join(dataset_path, "valid", "_annotations_bbox.coco.json"))

    trainer = Trainer(model, optimizer, device)
    num_epochs = 10
    trainer.train(train_loader,valid_loader, coco_gt, num_epochs)

if __name__ == "__main__":
    train()
