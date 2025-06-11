import os
import argparse
import torch
from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from utils.convert_labels import convert_labels
from data.augmentations import get_transforms
from data.dataloader import get_dataloaders
from trainer import Trainer

def main(args):
    # Step 1: Convert labels and prepare data
    print("Step 1: Converting labels and preparing data...")
    convert_labels(args.input_dir, args.output_dir)
    
    # Step 2: Setup MONAI transforms
    print("Step 2: Setting up MONAI transforms...")
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=["image", "label"], roi_size=args.img_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Step 3: Create dataloaders
    print("Step 3: Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.output_dir,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        val_transforms=val_transforms
    )
    
    # Step 4: Initialize model
    print("Step 4: Initializing model...")
    model = SwinUNETR(
        img_size=args.img_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint
    )
    
    # Step 5: Train model
    print("Step 5: Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR V2 Training Pipeline")
    
    # Data parameters
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=96, help="Input image size")
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels (4 for BraTS)")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels (3 for BraTS)")
    parser.add_argument("--feature_size", type=int, default=12, help="Feature size")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument("--dropout_path_rate", type=float, default=0.0, help="Dropout path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use checkpoint for memory efficiency")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    main(args) 