# run.py
import sys
import os
from kaggle_setup import setup_kaggle_notebook
from main import main
import argparse
import torch
import warnings

# CLI argument parsing for key parameters
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brats2023', choices=['brats2021', 'brats2023'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of data loader workers')
    parser.add_argument('--img_size', type=int, default=96, help='Input image size')
    parser.add_argument('--feature_size', type=int, default=48, help='Model feature size')
    parser.add_argument('--loss_type', type=str, default='hybrid', choices=['hybrid', 'dice'], help='Loss function: hybrid (DiceCE+Focal) or dice (Dice only)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for LR scheduler')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for loss (default: False)')
    parser.add_argument('--use_modality_attention', action='store_true', help='Enable Modality Attention module (default: False)')
    parser.add_argument('--overlap', type=float, default=0.7, help='Sliding window inference overlap (default: 0.5)')
    parser.add_argument('--use_tta', action='store_true', help='Enable Test Time Augmentation for validation (default: False)')
    return parser.parse_args()

cli_args = parse_cli_args()

"""
# Example usage:
!python kaggle_run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 2 \
  --num_workers 3 \
  --img_size 96 \
  --feature_size 48 \
  --overlap 0.7 \
  --loss_type dice \
  --learning_rate 1e-4 \
  --warmup_epochs 10 \
  --use_class_weights \
  --use_modality_attention \
  --use_tta
#
# Omit --use_class_weights, --use_modality_attention, and --use_tta to disable them (they are store_true flags)
"""

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook(cli_args.dataset)
print(f"Dataset prepared in: {output_dir}")

# Experimental Configuration
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=cli_args.batch_size,
    num_workers=cli_args.num_workers,
    pin_memory=True,
    persistent_workers=False,
    dataset=cli_args.dataset,  # Use the selected dataset
    
    # Model parameters
    img_size=cli_args.img_size,
    in_channels=4,
    out_channels=3,
    feature_size=cli_args.feature_size,

    # Training parameters
    learning_rate=cli_args.learning_rate,  # Now from CLI
    weight_decay=1e-5,
    warmup_epochs=cli_args.warmup_epochs,
    epochs=cli_args.epochs,
    accelerator='gpu',
    devices='auto',
    precision='16-mixed',
    strategy="ddp",
    log_every_n_steps=1,
    enable_checkpointing=True,
    benchmark=True,
    profiler="simple",
    use_amp=True,  # Enable mixed precision
    gradient_clip_val=1.0,
    use_v2=True,
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    downsample="mergingv2",
    
    # Enhanced model options
    use_class_weights=cli_args.use_class_weights,
    use_modality_attention=cli_args.use_modality_attention,
    use_tta=cli_args.use_tta,
    
    # Loss and training configuration
    class_weights=(1.0, 3.0, 5.0),  # Background, WT, TC, ET
    dice_ce_weight=0.6,
    focal_weight=0.4,
    threshold=0.5,
    optimizer_betas=(0.9, 0.999),
    optimizer_eps=1e-8,
    
    # Validation settings
    val_interval=1,
    save_interval=1,
    early_stopping_patience=15,
    limit_val_batches=5,  # Reduced for memory efficiency
    
    # Inference parameters
    roi_size=[96, 96, 96],  # Reduced ROI size
    sw_batch_size=1,
    overlap=cli_args.overlap,
    loss_type=cli_args.loss_type,
)

# Print final configuration summary
print("\n=== 🚀 OPTIMIZED SWINUNETR CONFIGURATION ===")
print(f"🎯 Batch size: {args.batch_size}")
print(f"📐 Image size: {args.img_size}")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔥 Warmup epochs: {args.warmup_epochs}")
print(f"🧮 Loss type: {args.loss_type}")
print(f"🔄 SW batch size: {args.sw_batch_size}")
print(f"📊 Total epochs: {args.epochs}")
print(f"🗂️ Dataset: {args.dataset}")
print(f"🏋️ Use class weights: {args.use_class_weights}")
print(f"🧠 Use modality attention: {args.use_modality_attention}")
print(f"🔄 Use TTA: {args.use_tta}")

def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        main(args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise e
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise e

# Start optimized training
if __name__ == "__main__":
    run_with_error_handling()