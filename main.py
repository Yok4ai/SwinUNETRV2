import os
import argparse
import torch
from monai.data import DataLoader, decollate_batch
import warnings

from src.data.augmentations import get_transforms
from src.data.dataloader import get_dataloaders
from src.models.trainer import setup_training, train_model

# Suppress warnings
warnings.filterwarnings('ignore')

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    """
    Main training function that orchestrates the entire training process
    
    Args:
        args: Argument namespace containing all configuration parameters
    """

    # Step 1: Setup MONAI transforms
    print("Step 1: Setting up MONAI transforms...")
    train_transforms, val_transforms = get_transforms(img_size=args.img_size, dataset=args.dataset)
    
    # Step 2: Create dataloaders
    print("Step 2: Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.input_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms
    )
    
    # Step 3: Setup training and initialize model
    print("Step 3: Setting up training...")
    model, trainer = setup_training(
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )
    # Print model parameter count
    total_params = count_parameters(model)
    print(f"\nModel Parameter Count: {total_params/1e6:.2f}M")
    print(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB\n")
  
    # Step 4: Train model
    print("Step 4: Starting training...")
    train_model(model, trainer, train_loader, val_loader)
