import os
import argparse
import torch
from monai.data import DataLoader, decollate_batch

from swinunetrv2.data.augmentations import get_transforms
from swinunetrv2.data.dataloader import get_dataloaders
from swinunetrv2.models.trainer import setup_training, train_model

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # Step 1: Setup MONAI transforms
    print("Step 1: Setting up MONAI transforms...")
    train_transforms, val_transforms = get_transforms(img_size=args.img_size)
    
    # Step 2: Create dataloaders
    print("Step 2: Creating dataloaders...")
    train_ds, val_ds = get_dataloaders(
        data_dir=args.input_dir,
        batch_size=args.batch_size,
        train_transforms=train_transforms,
        val_transforms=val_transforms
    )
    
    # Step 3: Setup training and initialize model
    print("Step 3: Setting up training...")
    model, trainer, train_loader, val_loader = setup_training(
        train_ds=train_ds,
        val_ds=val_ds,
        args=args
    )
    
    # Print model parameter count
    total_params = count_parameters(model)
    print(f"\nModel Parameter Count: {total_params/1e6:.2f}M")
    print(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB\n")
    
    # Step 4: Train model
    print("Step 4: Starting training...")
    train_model(model, trainer, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinUNETR V2 Training Pipeline")
    
    # Data parameters
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Enable pin memory for faster GPU transfer")
    parser.add_argument("--persistent_workers", action="store_true", default=False, help="Keep workers alive between epochs")
    
    # Model parameters
    parser.add_argument("--img_size", type=int, default=64, help="Input image size")
    parser.add_argument("--in_channels", type=int, default=4, help="Number of input channels (4 for BraTS)")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels (3 for BraTS)")
    parser.add_argument("--feature_size", type=int, default=24, help="Feature size")
    parser.add_argument("--depths", type=int, nargs='+', default=[1, 1, 1, 1], help="Depths of each stage")
    parser.add_argument("--num_heads", type=int, nargs='+', default=[3, 6, 12, 24], help="Number of attention heads")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--attn_drop_rate", type=float, default=0.1, help="Attention dropout rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use checkpoint for memory efficiency")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    
    # Validation settings
    parser.add_argument("--val_interval", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Model save interval in epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Early stopping patience")
    
    # Inference parameters
    parser.add_argument("--roi_size", type=int, nargs='+', default=[96, 96, 96], help="ROI size for sliding window inference")
    parser.add_argument("--sw_batch_size", type=int, default=2, help="Sliding window batch size")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap for sliding window inference")
    
    args = parser.parse_args()
    main(args) 