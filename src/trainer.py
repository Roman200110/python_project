import torch
import numpy as np
import pandas as pd
import os
import math
from tqdm import tqdm
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.utils import save_checkpoint

class Trainer:
    """
    A class to represent the training process.

    Attributes
    ----------
    model : torch.nn.Module
        the neural network model
    optimizer : torch.optim.Optimizer
        the optimizer
    device : torch.device
        the device to run the model on (CPU or GPU)
    checkpoint_dir : str
        directory to save checkpoints

    Methods
    -------
    train_one_epoch(loader, epoch):
        Trains the model for one epoch.
    validate(loader):
        Validates the model on the validation set.
    train(train_loader, valid_loader, num_epochs):
        Trains the model for the given number of epochs.
    save_checkpoint(epoch):
        Saves the model checkpoint.
    """

    def __init__(self, model, optimizer, device, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, loader, epoch):
        """Trains the model for one epoch."""
        self.model.train()
        all_losses = []
        all_losses_dict = []

        for images, targets in tqdm(loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            all_losses.append(loss_value)
            all_losses_dict.append({k: v.item() for k, v in loss_dict.items()})

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        all_losses_dict = pd.DataFrame(all_losses_dict)
        print(f"Epoch {epoch}, Loss: {np.mean(all_losses):.4f}")
        print(all_losses_dict.mean())

    def validate(self, loader, coco_gt):
        """Validates the model on the validation set."""
        self.model.eval()
        results = []
        img_ids = set(coco_gt.getImgIds())
        with torch.no_grad():
            for images, targets in tqdm(loader):
                images = list(image.to(self.device) for image in images)
                outputs = self.model(images)

                for i, output in enumerate(outputs):
                    image_id = targets[i]["image_id"].item()
                    # if image_id not in img_ids:
                    #     continue
                    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                        bbox = box.tolist()
                        # Convert to COCO format: [x_min, y_min, width, height]
                        bbox[2] -= bbox[0]
                        bbox[3] -= bbox[1]
                        results.append({
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": bbox,
                            "score": score.item()
                        })

        # Load results into COCO API
        coco_dt = coco_gt.loadRes(results)

        # Evaluate the results
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.iouThrs = np.array([0.5])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # print(f"mAP@0.5: {coco_eval.stats[1]:.4f}")
        return coco_eval.stats[1]

    def train(self, train_loader, valid_loader, coco_gt, num_epochs):
        """Trains the model for the given number of epochs."""
        for epoch in range(num_epochs):
            self.train_one_epoch(train_loader, epoch)

            if (epoch + 1) % 2 == 0:
                print(f"Validating at epoch {epoch + 1}")
                self.validate(valid_loader, coco_gt)

            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        """Saves the model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        save_checkpoint(state, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')
