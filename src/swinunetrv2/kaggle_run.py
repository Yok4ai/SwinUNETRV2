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

def estimate_parameters(args):
    """Estimate model parameters"""
    # Rough estimation for SwinUNETR with given parameters
    embed_dim = args.embed_dim
    depths = args.depths
    
    # Encoder parameters (rough estimate)
    encoder_params = embed_dim * embed_dim * 8  # Patch embedding
    for i, depth in enumerate(depths):
        layer_dim = embed_dim * (2 ** i)
        encoder_params += depth * layer_dim * layer_dim * 12  # Attention + MLP
    
    # Decoder parameters
    decoder_params = args.decoder_embed_dim * sum([embed_dim * (2**i) for i in range(len(depths))])
    decoder_params += args.decoder_embed_dim * args.decoder_embed_dim * 4
    
    return encoder_params + decoder_params

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
    img_size=128,
    in_channels=4,
    out_channels=3,
    feature_size=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    patch_size=4,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    use_checkpoint=True,
    
    # Training parameters
    learning_rate=1e-4,  # More conservative
    weight_decay=1e-5,
    epochs=50,  # Increased epochs
    warmup_epochs=5,
    device='cuda',
    use_amp=True,
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,  # Increased for effective batch size
    
    # Validation settings
    val_interval=1,
    save_interval=1,
    early_stopping_patience=15,
    limit_val_batches=10,  # Increased for better validation
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.25,
)

# Enhanced configuration validation
def validate_improved_config(args):
    """Enhanced validation for improved configuration"""
    print("\n🔍 Validating optimized configuration...")
    
    # Basic validations
    assert args.embed_dim == args.feature_size, f"embed_dim ({args.embed_dim}) should match feature_size ({args.feature_size})"
    assert len(args.depths) == len(args.num_heads), "depths and num_heads must have same length"
    assert args.window_size <= args.img_size // 4, f"window_size ({args.window_size}) too large for img_size ({args.img_size})"
    assert args.patch_size <= args.img_size // 8, f"patch_size ({args.patch_size}) too large for img_size ({args.img_size})"
    
    # Memory estimations
    estimated_params = estimate_parameters(args)
    print(f"📊 Estimated parameters: {estimated_params/1e6:.1f}M")
    
    # Memory check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔧 Available GPU memory: {gpu_memory:.1f} GB")
        
        # Rough memory estimation
        estimated_memory = args.batch_size * 1.2  # GB per batch item
        if estimated_memory > gpu_memory * 0.8:
            print(f"⚠️  Warning: Estimated memory usage ({estimated_memory:.1f}GB) may exceed available memory")
            print(f"   Consider reducing batch_size from {args.batch_size} to {int(args.batch_size * 0.7)}")
    
    print("✅ Optimized configuration validation passed!")

# Validate configuration
validate_improved_config(args)

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
        print("\n🚀 Starting training with optimized configuration...")
        print("Expected improvements:")
        print("  • Better model checkpointing")
        print("  • Improved validation metrics")
        print("  • More stable training")
        print("  • Better memory management")
        
        main(args)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print("🔧 Suggested fixes:")
            print(f"1. Reduce batch_size from {args.batch_size} to {args.batch_size//2}")
            print("2. Reduce accumulate_grad_batches from {args.accumulate_grad_batches} to {args.accumulate_grad_batches//2}")
            print("3. Reduce sw_batch_size from {args.sw_batch_size} to 1")
            print("4. Reduce decoder_embed_dim from {args.decoder_embed_dim} to 192")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Suggested fixes:")
        print("1. Make sure all required packages are installed")
        print("2. Check for version conflicts")
        print("3. Verify the package structure")
        raise e
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 General troubleshooting:")
        print("1. Check GPU memory usage")
        print("2. Verify dataset integrity")
        print("3. Check file permissions")
        print("4. Monitor system resources")
        raise e

# Start optimized training
if __name__ == "__main__":
    run_with_error_handling()