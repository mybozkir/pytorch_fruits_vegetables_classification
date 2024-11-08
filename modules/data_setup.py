"""
Contains custom data setup functions for PyTorch projects.
"""
# Libraries
import os
import torch
import torchvision

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from typing import Tuple, List, Dict


def createSingleDataloader(dir_path: Path,
                           batch_size: int,
                           transform: torchvision.transforms = None,
                           shuffle: bool = False,
                           target_transform: torchvision.transforms = None,
                           num_workers: int = os.cpu_count(),
                           class_list: bool = False,
                           class_dict: bool = False) -> Tuple[DataLoader, List[str], Dict[str, int]]:
    """Creates a single dataloader by using directory path. Requires torchvision.

    This function creates DataLoader from a given path, creating an ImageFolder from
    the path firstful.
    
    Args:
        dir_path (Path): Path of data directory to be used for DataLoader creation.
        batch_size (int): Batch size to load.
        transform (torchvision.transforms, None): Transformation object to be applied. Defaults None.
        shuffle (bool, optional): Required True, if data needs to be shuffled. Defaults False.
        target_transform (torchvision.transforms, optional): Target transformation object if necessary. Defaults None.
        num_workers (int, optional): Number of workers for CPU. Defaults os.cpu_count()
        class_dict (bool, optional): True, if class dictionary necessary for output. Defaults False.
    
    Returns:
        DataLoader (torch.utils.data.DataLoader), class list (list), class dictionary (dict)
    """
    
    # Create single Image Folder
    imageFolder = ImageFolder(root = dir_path,
                              transform = transform,
                              target_transform = target_transform)
    
    # Create class list and dictionary.
    classList = imageFolder.classes
    classDict = imageFolder.class_to_idx
    
    # Create single DataLoader
    dataLoader = DataLoader(dataset = imageFolder,
                            batch_size = batch_size,
                            num_workers = num_workers,
                            shuffle = shuffle,
                            pin_memory = True)
    
   


    # Return results
    if (class_list == True) & (class_dict == True):
         # Print the DataLoader [INFO]
        print(f"[INFO] {len(dataLoader)} data loaders created with batch size {batch_size} from {dir_path}")
        return dataLoader, classList, classDict
    elif (class_list == True) & (class_dict == False):
        # Print the DataLoader [INFO]
        print(f"[INFO] {len(dataLoader)} data loaders created with batch size {batch_size} from {dir_path}")
        return dataLoader, classList
    elif (class_list == False) & (class_dict == True):
        # Print the DataLoader [INFO]
        print(f"[INFO] {len(dataLoader)} data loaders created with batch size {batch_size} from {dir_path}")
        return dataLoader, classDict
    else:
        # Print the DataLoader [INFO]
        print(f"[INFO] {len(dataLoader)} data loaders created with batch size {batch_size} from {dir_path}")
        return dataLoader
