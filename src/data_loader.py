import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import copy
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    """
    A custom dataset class for loading data from the COCO format.

    Attributes
    ----------
    root : str
        path to the dataset root directory
    split : str
        dataset split (train, valid, or test)
    transforms : A.Compose
        albumentations transforms to be applied on the images and annotations

    Methods
    -------
    _load_image(id):
        Loads an image given its id.
    _load_target(id):
        Loads the target (annotations) given the image id.
    __getitem__(index):
        Returns a transformed image and its corresponding target.
    __len__():
        Returns the number of samples in the dataset.
    """
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.coco = COCO(os.path.join(root, split, "_annotations_bbox.coco.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [id for id in self.ids if len(self._load_target(id)) > 0]

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, self.split, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = copy.deepcopy(target)

        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']

        new_boxes = []
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['image_id'] = torch.tensor([id])
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)

        return image.div(255), targ

    def __len__(self):
        return len(self.ids)

def get_transforms(train=False):
    """
    Returns the data augmentation transforms to be applied on the dataset.

    Parameters
    ----------
    train : bool
        whether the transformations are for the training dataset

    Returns
    -------
    A.Compose
        the composed albumentations transforms
    """
    if train:
        transform = A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(640, 640),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

def collate_fn(batch):
    """Combines a list of samples to form a mini-batch of Tensor(s)."""
    return tuple(zip(*batch))

def create_dataloaders(dataset_path, batch_size=4, num_workers=4):
    """
    Creates data loaders for the train, valid, and test datasets.

    Parameters
    ----------
    dataset_path : str
        path to the dataset root directory
    batch_size : int
        number of samples per batch
    num_workers : int
        number of subprocesses to use for data loading

    Returns
    -------
    tuple
        train, validation, and test data loaders
    """
    train_dataset = CustomDataset(root=dataset_path, split='train', transforms=get_transforms(True))
    valid_dataset = CustomDataset(root=dataset_path, split='valid', transforms=get_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, valid_loader
