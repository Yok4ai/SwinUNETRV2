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


# 🎯 FIXED CONFIGURATION FOR BETTER PERFORMANCE
args = argparse.Namespace(
    # Data parameters - keeping your working values
    input_dir='/kaggle/working',
    batch_size=4,  # FIXED: Reduced batch size for stability
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,  # Disabled to fix pickling error
    
    # 🚀 FIXED MODEL PARAMETERS (Standard SwinUNETR)
    img_size=128,
    in_channels=4,  # 4 modalities for BraTS
    out_channels=3,  # 3 tumor regions
    feature_size=96,  # FIXED: Standard SwinUNETR dimension
    embed_dim=96,    # FIXED: Standard SwinUNETR embedding dimension
    depths=[2, 2, 6, 2],  # Keep proven architecture
    num_heads=[3, 6, 12, 24],  # FIXED: Proper head scaling
    window_size=7,    # FIXED: Standard window size
    mlp_ratio=4.0,    # FIXED: Standard MLP ratio
    decoder_embed_dim=256,  # FIXED: Larger decoder for better representation
    patch_size=4,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    use_checkpoint=True,  # Enable checkpointing for memory efficiency
    
    # 📈 FIXED TRAINING PARAMETERS
    learning_rate=5e-4,  # FIXED: More conservative learning rate
    weight_decay=1e-5,   # FIXED: Reduced weight decay
    epochs=30,  # Keep 30 epochs
    warmup_epochs=5,  # FIXED: Reduced warmup
    device='cuda',
    use_amp=True,
    gradient_clip_val=1.0,
    accumulate_grad_batches=3,  # FIXED: Increased accumulation (effective batch = 12)
    
    # Enhanced validation settings
    val_interval=1,
    save_interval=3,
    early_stopping_patience=10,
    limit_val_batches=5,  # Faster validation
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.25,
)

# Enhanced configuration validation
def validate_improved_config(args):
    """Enhanced validation for improved configuration"""
    print("\n🔍 Validating fixed configuration...")
    
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
        
        # Rough memory estimation (very approximate)
        estimated_memory = args.batch_size * 1.2  # GB per batch item (larger model)
        if estimated_memory > gpu_memory * 0.8:
            print(f"⚠️  Warning: Estimated memory usage ({estimated_memory:.1f}GB) may exceed available memory")
            print(f"   Consider reducing batch_size from {args.batch_size} to {int(args.batch_size * 0.7)}")
    
    print("✅ Fixed configuration validation passed!")


# Validate configuration
validate_improved_config(args)

# Print final configuration summary
print("\n=== 🚀 FIXED LIGHTWEIGHT SWINUNETR CONFIGURATION ===")
print(f"🎯 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
print(f"📐 Image size: {args.img_size}")
print(f"🧠 Embed dim: {args.embed_dim} (FIXED: Standard SwinUNETR)")
print(f"🏗️  Depths: {args.depths}")
print(f"👁️  Num heads: {args.num_heads} (FIXED: Proper scaling)")
print(f"🪟 Window size: {args.window_size} (FIXED: Standard)")
print(f"🔧 Decoder embed dim: {args.decoder_embed_dim} (FIXED: Larger)")
print(f"⚡ Learning rate: {args.learning_rate} (FIXED: Conservative)")
print(f"🔄 SW batch size: {args.sw_batch_size}")
print(f"🕐 Warmup epochs: {args.warmup_epochs}")
print(f"📊 Total epochs: {args.epochs}")

def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        print("\n🚀 Starting training with FIXED configuration...")
        print("Expected improvements:")
        print("  • Better TC segmentation (major issue fixed)")
        print("  • Improved ET detection")
        print("  • More stable training")
        print("  • Faster convergence")
        
        main(args)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print("🔧 Suggested fixes:")
            print(f"1. Reduce batch_size from {args.batch_size} to {args.batch_size//2}")
            print("2. Reduce accumulate_grad_batches from 3 to 2")
            print("3. Reduce sw_batch_size from 2 to 1")
            print("4. Reduce decoder_embed_dim from 256 to 192")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Suggested fixes:")
        print("1. Make sure the fixed architecture.py is in your swinunetrv2 package")
        print("2. Update your main.py to support the fixed model")
        print("3. Check that all required dependencies are installed")
        raise e
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 General troubleshooting:")
        print("1. Check that your swinunetrv2 package supports the fixed model")
        print("2. Verify all configuration parameters are valid")
        print("3. Make sure dataset is properly formatted")
        print("4. Check file permissions and disk space")
        raise e

# Start fixed training
if __name__ == "__main__":
    run_with_error_handling()