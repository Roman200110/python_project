import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as TorchDataLoader, random_split


class DataLoader:
    """
    A class used to load and preprocess data for the animal classification project.

    Attributes
    ----------
    data_dir : str
        The directory where the dataset is stored.
    transform : torchvision.transforms.Compose
        The transformations to apply to the images.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation set.

    Methods
    -------
    load_data(batch_size):
        Loads and splits the dataset into training and validation sets.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_loader = None
        self.val_loader = None

    def load_data(self, batch_size: int = 32):
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return self.train_loader, self.val_loader
