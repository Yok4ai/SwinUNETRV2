import json
import os
from monai.data import Dataset, DataLoader

def get_dataloaders(data_dir, batch_size, num_workers, train_transforms, val_transforms):
    """
    Create training and validation DataLoaders.
    
    Args:
        data_dir (str): Directory containing the dataset.json file
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for DataLoader
        train_transforms: Transformations for training data
        val_transforms: Transformations for validation data
    
    Returns:
        tuple: (train_loader, val_loader)
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

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader 