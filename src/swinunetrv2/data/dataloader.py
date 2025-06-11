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

# Load dataset
dataset_path = "/kaggle/working/dataset.json"
with open(dataset_path) as f:
    datalist = json.load(f)["training"]

# Split dataset into training (80%) and validation (20%)
train_files, val_files = train_test_split(datalist, test_size=0.2, random_state=42)

### For quick iterations use 10 percent dataset

# train_files = train_files[:int(len(train_files) * 0.3)]  # 10% of training data
# val_files = val_files[:int(len(val_files) * 0.3)]  # 10% of validation data

# Set deterministic behavior
set_determinism(seed=0)

cache_dir = "/kaggle/working/cache"


train_transform = train_transform
val_transform = val_transform

# Create MONAI datasets

train_ds = Dataset(data=train_files, transform=train_transform)
# train_ds = PersistentDataset(data=train_files, transform=train_transform, cache_dir=cache_dir)


val_ds = Dataset(data=val_files, transform=val_transform)
# val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=0.5, num_workers=4)
# val_ds = PersistentDataset(data=val_files, transform=val_transform, cache_dir=cache_dir)



# Dataloaders
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=False)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=False)