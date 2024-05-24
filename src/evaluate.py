import torch
import os
import json
from tqdm import tqdm
from torchvision import transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data_loader import create_dataloaders
from model import AnimalClassifier
from utils import load_checkpoint
import cv2
import numpy as np
from src.data_loader import create_dataloaders, CustomDataset, get_transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.trainer import Trainer



def collate_fn(batch):
    """Combines a list of samples to form a mini-batch of Tensor(s)."""
    return tuple(zip(*batch))

def evaluate():
    """
    The main function to run the evaluation process.
    """
    # Configuration
    dataset_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/data/raw/TAWIRI 02.v3i.coco"  # Update this to your dataset path
    checkpoint_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/checkpoints/model_epoch_20.pth"  # Update this to the actual checkpoint path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = 'test'
    # Load the ground truth COCO annotations
    ann_file = os.path.join(dataset_path, split, '_annotations_bbox.coco.json')
    coco_gt = COCO(ann_file)

    test_dataset = CustomDataset(root=dataset_path, split='test', transforms=get_transforms(False))

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load model
    num_classes = len(coco_gt.cats.keys())
    model = AnimalClassifier(num_classes).get_model().to(device)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)

    load_checkpoint(checkpoint_path, model, optimizer)
    trainer = Trainer(model, optimizer, device)

    trainer.validate(test_loader, coco_gt)



if __name__ == "__main__":
    evaluate()