"""
Contains train, test and evaluation functions for PyTorch computer-vision projects.
"""

# Libraries
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as F

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict
from tqdm.auto import tqdm

# Set device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device) -> Tuple[float, float]:
    
    # Take the model in train mode
    model.train()

    # Set the initial values of train_loss and train_acc
    train_loss, train_acc = 0, 0

    # Loop through batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        train_pred_logits = model(X)
        train_pred_probs = torch.softmax(train_pred_logits, dim = 1)
        train_pred_labels = torch.argmax(train_pred_probs, dim = 1)

        # 2. Calculate loss and accuracy
        loss = loss_fn(train_pred_logits, y)
        train_loss += loss.item()

        acc = (train_pred_labels == y).sum().item() / len(train_pred_logits)
        train_acc += acc

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
    
    # Adjust the train_loss and train_acc according to batch size
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = device) -> Tuple[float, float]:
    # Take the model in eval mode
    model.eval()

    # Set the initial values of test_loss and test_acc
    test_loss, test_acc = 0, 0
    
    # Open context manager
    with torch.inference_mode():
        
        # Loop through batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)
            test_pred_probs = torch.softmax(test_pred_logits, dim = 1)
            test_pred_labels = torch.argmax(test_pred_probs, dim = 1)

            # 2. Calculate loss and acc
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            acc = (test_pred_labels == y).sum().item() / len(test_pred_logits)
            test_acc += acc

        
    # 3. Adjust metrics according to batch size
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train_eval(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               epochs: int,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device = device) -> Dict[str, List[float]]:
    
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                       dataloader = train_dataloader,
                                       optimizer = optimizer,
                                       loss_fn = loss_fn,
                                       device = device)
    
        test_loss, test_acc = test_step(model = model,
                                    dataloader = test_dataloader,
                                    loss_fn = loss_fn,
                                    device = device)
        
        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Train Accuracy: {train_acc:.3f} | "
            f"Test Loss: {test_loss:.3f} | "
            f"Test Accuracy: {test_acc:.3f} | "
        )
    
        # Add the results in results dictionary
        results['train_loss'].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results['train_acc'].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results['test_loss'].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results['test_acc'].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    
    return results
