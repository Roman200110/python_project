import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    A class used to train and evaluate the animal classification model.

    Attributes
    ----------
    model : AnimalClassifier
        The model to be trained.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation set.

    Methods
    -------
    train(epochs):
        Trains the model for a specified number of epochs.
    evaluate():
        Evaluates the model's performance on the validation set.
    """

    def __init__(self, model, train_loader, val_loader, lr: float = 0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
        return accuracy
