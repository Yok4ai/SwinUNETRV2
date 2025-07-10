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
    return parser.parse_args()

cli_args = parse_cli_args()

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
    learning_rate=1e-4,  # More conservative
    weight_decay=1e-5,
    warmup_epochs=30,
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
    use_enhanced_model=True,  # Enable enhanced model for experimentation
    use_modality_attention=True,
    use_class_weights=True,
    
    # Validation settings
    val_interval=1,
    save_interval=1,
    early_stopping_patience=15,
    limit_val_batches=5,  # Reduced for memory efficiency
    
    # Inference parameters
    roi_size=[96, 96, 96],  # Reduced ROI size
    sw_batch_size=1,
    overlap=0.25,
)

# Print final configuration summary
print("\n=== 🚀 OPTIMIZED SWINUNETR CONFIGURATION ===")
print(f"🎯 Batch size: {args.batch_size}")
print(f"📐 Image size: {args.img_size}")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔄 SW batch size: {args.sw_batch_size}")
print(f"📊 Total epochs: {args.epochs}")
print(f"🗂️ Dataset: {args.dataset}")
print(f"🏋️ Use class weights: {args.use_class_weights}")

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