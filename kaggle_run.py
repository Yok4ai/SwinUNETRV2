# run.py
import sys
import os
from kaggle_setup import setup_kaggle_notebook
from main import main
import argparse
import torch
import warnings

def optimize_gpu_usage():
    """Optimize GPU memory usage and performance"""
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        print("✅ GPU optimizations applied")

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Apply enhanced GPU optimizations
# optimize_gpu_usage()

# Experimental Configuration
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=2,
    num_workers=3,
    pin_memory=True,
    persistent_workers=False,
    dataset="brats2021",  # Added dataset option
    
    # Model parameters
    img_size=96,
    in_channels=3,
    out_channels=3,
    feature_size=48,

    # Training parameters
    learning_rate=1e-4,  # More conservative
    weight_decay=1e-5,
    epochs=5,
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
    use_mlp_decoder=True,
    mlp_hidden_ratio=4,
    dropout_rate=0.1,
    
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