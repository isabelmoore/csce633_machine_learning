import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# todo
def train_epoch(model, device, train_loader, optimizer, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to use (cuda or cpu).
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (nn.Module): The loss function.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # Set the model to training mode
    running_loss = 0.0

    # todo: Implement the training loop:
    # Perform a forward pass + backward pass, update model weights and running_loss
    print("todo: Implement train_epoch")

    return running_loss


# todo
def validate_epoch(model, device, val_loader, criterion):
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to use (cuda or cpu).
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: (average validation loss, validation accuracy)
    """
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    val_acc = 0.0

    # todo: Implement the validation loop:
    # Calculate the validation loss and compute validation accuracy
    # Tip: when using the model, you should disable gradient calculation
    # using with torch.no_grad():
    print("todo: Implement validate_epoch")
    return val_loss, val_acc

# todo
def plot_results(train_losses, val_losses, val_metrics):
    """
    Plots the training and validation metrics.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        val_metrics (list): List of validation metrics per epoch.
    """
    # TODO: Implement plotting
    # Create two subplots:
    # 1. Loss (Training vs. Validation) vs. Epoch
    # 2. Metric (Validation) vs. Epoch
    print("TODO: Implement plot_results")
    
    
# --- Setup ---
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
epochs = 15
batch_size = 128


# todo
# --- Load Data ---
normalize = transforms.Normalize((0.1307,), (0.3081,))      # MNIST-specific
train_tfms = transforms.Compose([transforms.ToTensor(), normalize])
test_tfms = transforms.Compose([transforms.ToTensor(), normalize])

# Datasets
full_train_set = datasets.MNIST("./data", train=True, download=True,
                                transform=train_tfms)
test_set = datasets.MNIST("./data", train=False, download=True,
                          transform=test_tfms)
##### todo #####
# Split full_train_set to train_set and val_set
train_set, val_set = None, None
################
num_workers = 2 if use_cuda else 0
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# Initialize the model
def build_resnet18_mnist(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
model = build_resnet18_mnist().to(device)
# Define the loss
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# Create lists to store history
train_losses, val_losses, val_accuracies = [], [], []


# todo
# Training loop
for epoch in range(epochs):
    ##### todo #####
    # Call train_epoch and validate_epoch
    # Store the results, and keep track of best model
    pass
    ################
    # Update learning rate
    if scheduler is not None:
        scheduler.step()

##### todo #####
# Plot training and validation curves
# Compute testing accuracy
################




##### PART 2 #####
import os
import warnings
warnings.filterwarnings("ignore")
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.sampler import DualSampler
import medmnist
from medmnist import ChestMNIST

# todo
def validate_epoch(model, device, val_loader, criterion):
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to use (cuda or cpu).
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: (average validation loss, validation auc score)
    """
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    val_auc = 0.0

    # todo: Implement the validation loop:
    # Calculate the validation loss and compute validation auc score
    # Tip: when using the model, you should disable gradient calculation
    # using with torch.no_grad():
    print("todo: Implement validate_epoch")
    return val_loss, val_auc

# --- Setup ---
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
epochs = 15
batch_size = 128

# --- Load Data ---
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

os.makedirs("./data", exist_ok=True)
train_set = ChestMNIST(root="./data", split="train", download=True,
                       transform=train_tfms, as_rgb=True)
val_set   = ChestMNIST(root="./data", split="val",   download=True,
                       transform=eval_tfms, as_rgb=True)
test_set  = ChestMNIST(root="./data", split="test",  download=True,
                       transform=eval_tfms, as_rgb=True)

target_idx = 1
for dataset in (train_set, val_set, test_set):
    dataset.labels = dataset.labels[:, target_idx]
num_workers = 2 if use_cuda else 0
train_set.targets = train_set.labels
sampler = DualSampler(train_set, batch_size=batch_size, sampling_rate=0.1)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                          num_workers=num_workers, pin_memory=True, sampler=sampler)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# Initialize the model
def build_resnet18_medmnist() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model
model = build_resnet18_medmnist().to(device)
# Define the loss
criterion = AUCMLoss()
# Define the optimizer
optimizer = PESG(model.parameters(), loss_fn=criterion, lr=0.1, momentum=0.9,
                 weight_decay=1e-4, epoch_decay=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# Create lists to store history
train_losses, val_losses, val_aucs = [], [], []

# todo
# Training loop
for epoch in range(epochs):
    ##### todo #####
    # Call train_epoch and validate_epoch
    # Store the results, and keep track of best model
    pass
    ################
    # Update learning rate
    if scheduler is not None:
        scheduler.step()

##### todo #####
# Plot training and validation curves
# Compute testing auc score
################