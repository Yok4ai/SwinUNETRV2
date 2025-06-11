import os
import argparse
import torch
from monai.data import DataLoader, decollate_batch

from swinunetrv2.utils.convert_labels import convert_labels
from swinunetrv2.data.augmentations import get_transforms
from swinunetrv2.data.dataloader import get_dataloaders
from swinunetrv2.models.trainer import setup_training, train_model

def main(args):
    # Step 1: Convert labels and prepare data
    print("Step 1: Converting labels and preparing data...")
    convert_labels(args.input_dir, args.output_dir)
    
    # Step 2: Setup MONAI transforms
    print("Step 2: Setting up MONAI transforms...")
    train_transforms, val_transforms = get_transforms(img_size=args.img_size)
    
    # Step 3: Create dataloaders
    print("Step 3: Creating dataloaders...")
    train_ds, val_ds = get_dataloaders(
        data_dir=args.output_dir,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        val_transforms=val_transforms
    )
    
    # Step 4: Setup training and initialize model
    print("Step 4: Setting up training...")
    model, trainer, train_loader, val_loader = setup_training(
        train_ds=train_ds,
        val_ds=val_ds,
        max_epochs=args.epochs
    )
    
    # Step 5: Train model
    print("Step 5: Starting training...")
    train_model(model, trainer, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR V2 Training Pipeline")
    
    # Data parameters
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=96, help="Input image size")
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels (4 for BraTS)")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels (3 for BraTS)")
    parser.add_argument("--feature_size", type=int, default=48, help="Feature size")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument("--dropout_path_rate", type=float, default=0.0, help="Dropout path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use checkpoint for memory efficiency")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    main(args) 