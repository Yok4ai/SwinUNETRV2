# run.py
from swinunetrv2.kaggle_setup import setup_kaggle_notebook
from swinunetrv2.main import main
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

# 🔥 ENHANCED GPU OPTIMIZATION SETTINGS
def optimize_gpu_usage():
    """Apply enhanced GPU optimizations for better performance"""
    
    # Enable cudNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable TensorFloat-32 for better performance on Ampere GPUs
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set memory management
    torch.cuda.empty_cache()
    
    # Additional memory optimizations
    if torch.cuda.is_available():
        # Set memory fraction to avoid fragmentation
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
    
    print("🔥 Enhanced GPU optimizations applied!")

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Apply enhanced GPU optimizations
optimize_gpu_usage()

# 🎯 OPTIMIZED CONFIGURATION FOR BETTER PERFORMANCE
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=2,  # Reduced for stability
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    
    # Model parameters
    img_size=64,
    in_channels=4,
    out_channels=3,
    feature_size=24,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    use_checkpoint=True,  # Enable gradient checkpointing
    
    # Training parameters
    learning_rate=1e-4,  # More conservative
    weight_decay=1e-5,
    epochs=5,
    warmup_epochs=5,
    device='cuda',
    use_amp=True,  # Enable mixed precision
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,  # Reduced to minimize memory usage
    
    # Validation settings
    val_interval=1,
    save_interval=1,
    early_stopping_patience=15,
    limit_val_batches=5,  # Reduced for memory efficiency
    
    # Inference parameters
    roi_size=[64, 64, 64],  # Reduced ROI size
    sw_batch_size=1,
    overlap=0.25,
)


# Print final configuration summary
print("\n=== 🚀 OPTIMIZED SWINUNETR CONFIGURATION ===")
print(f"🎯 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
print(f"📐 Image size: {args.img_size}")
print(f"👁️  Num heads: {args.num_heads}")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔄 SW batch size: {args.sw_batch_size}")
print(f"🕐 Warmup epochs: {args.warmup_epochs}")
print(f"📊 Total epochs: {args.epochs}")

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