import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # Return logits, not probabilities
    
def load_cnn_model(path="mnist_cnn.pt") -> Model | None:
    """
    Loads a model, however it loads the entire model, not just the weights.
    This is because we need the architecture to be saved as well in the a4s project.
    """
    model: Model
    try:
        model = torch.jit.load(path)
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None
    
    return model

def get_mnist_dataset_loaders():
    """
    :return train_loader: DataLoader for training set
    :return test_loader: DataLoader for test set
    """
    train_dataset, test_dataset = get_mnist_dataset()
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

def get_mnist_dataset():
    """
    :return train_dataset: training dataset
    :return test_dataset: test dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset