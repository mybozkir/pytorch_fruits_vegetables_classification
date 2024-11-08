"""
Custom functions for PyTorch computer-vision projects.
"""
# Libraries
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict

def openRandomImageFromPathList(imagePathList: List[os.PathLike],
                                seed: None = 42):
    if seed is not None:
        seed = seed
        random.seed(seed)

    randomImagePath = random.choice(imagePathList)
    randomImage = Image.open(randomImagePath)
    randomImageLabel = randomImagePath.parent.stem

    print(f"The class label of the image: {randomImageLabel}")
    print(f"The colour mode of the image: {randomImage.mode}")
    print(f"The size of the image: {randomImage.size}")
    print(f"The height of the image: {randomImage.height}")
    print(f"The width of the image: {randomImage.width}\n")

    return randomImage

def set_seeds(seed: int = 42):
    """Sets random seed for PyTorch operations. Requires torch library.
    
    Args:
        seed (int, optional): Random seed to set. Defaults 42 (the meaning of life).
    """
    # Set the seed for general PyTorch operations
    torch.manual_seed(seed)

    # Set the seed for GPU operations
    torch.cuda.manual_seed(seed)

