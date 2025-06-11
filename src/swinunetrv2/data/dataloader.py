import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd,
    Spacingd, RandSpatialCropd, RandFlipd, NormalizeIntensityd,
    RandScaleIntensityd, RandShiftIntensityd
)
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
import json
import os
from monai.data import load_decathlon_datalist

def get_dataloaders(data_dir, batch_size, train_transforms, val_transforms):
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir (str): Directory containing the dataset.json file
        batch_size (int): Batch size for training
        train_transforms: Training transforms
        val_transforms: Validation transforms
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Load dataset
    dataset_path = data_dir if data_dir.endswith('dataset.json') else os.path.join(data_dir, "dataset.json")
    with open(dataset_path) as f:
        datalist = json.load(f)["training"]
    
    # Split into train and validation (80/20 split)
    train_size = int(len(datalist) * 0.8)
    train_files = datalist[:train_size]
    val_files = datalist[train_size:]
    
    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    return train_ds, val_ds 