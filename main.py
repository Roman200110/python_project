# main.py
import torch
from src.data_loader import DataLoader
from src.model import AnimalClassifier
from src.trainer import Trainer

def main():
    data_dir = "data/raw/train_val2019"
    data_loader = DataLoader(data_dir)
    train_loader, val_loader = data_loader.load_data(batch_size=32)

    model = AnimalClassifier(num_classes=4)
    trainer = Trainer(model, train_loader, val_loader)

    trainer.train(epochs=10)
    trainer.evaluate()

if __name__ == "__main__":
    main()
