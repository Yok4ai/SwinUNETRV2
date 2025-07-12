import json
import os
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
    
    # Randomly split into train and validation (80/20 split)
    train_files, val_files = train_test_split(datalist, test_size=0.2, random_state=42)

    # Create datasets
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader 