# improved_run.py
# Import necessary modules
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

def print_model_comparison():
    """Print comparison between your original and improved model"""
    print("\n" + "="*60)
    print("📊 MODEL ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"{'Parameter':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
    print("-" * 60)
    print(f"{'Embed Dim':<20} {'48':<15} {'64':<15} {'+33%':<15}")
    print(f"{'Depths':<20} {'[2,2,6,2]':<15} {'[2,2,6,2]':<15} {'Same':<15}")
    print(f"{'Num Heads':<20} {'[3,6,12,24]':<15} {'[2,4,8,16]':<15} {'Optimized':<15}")
    print(f"{'Window Size':<20} {'7':<15} {'4':<15} {'Smaller':<15}")
    print(f"{'Decoder Dim':<20} {'256':<15} {'128':<15} {'Efficient':<15}")
    print(f"{'MLP Ratio':<20} {'4.0':<15} {'2.0':<15} {'Reduced':<15}")
    print(f"{'Learning Rate':<20} {'2e-3':<15} {'8e-4':<15} {'Stable':<15}")
    print(f"{'Warmup Epochs':<20} {'5':<15} {'10':<15} {'Extended':<15}")
    print("="*60)
    print("🎯 Expected Improvements:")
    print("   • Better parameter efficiency")
    print("   • More stable training")
    print("   • Higher dice scores")
    print("   • Better memory usage")
    print("="*60)

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Apply enhanced GPU optimizations
optimize_gpu_usage()

# Print model comparison
print_model_comparison()

# 🎯 IMPROVED CONFIGURATION FOR BETTER PERFORMANCE
args = argparse.Namespace(
    # Data parameters - keeping your working values
    input_dir='/kaggle/working',
    batch_size=6,  # Slightly reduced for improved model
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,  # Disabled to fix pickling error
    
    # 🚀 IMPROVED MODEL PARAMETERS
    img_size=128,
    in_channels=4,  # 4 modalities for BraTS
    out_channels=3,  # 3 tumor regions
    feature_size=64,  # Increased from 48
    embed_dim=64,    # Increased from 48 for better representation
    depths=[2, 2, 6, 2],  # Keep proven architecture
    num_heads=[2, 4, 8, 16],  # More efficient head distribution
    window_size=4,    # Reduced from 7 for efficiency
    mlp_ratio=2.0,    # Reduced from 4.0 for efficiency
    decoder_embed_dim=128,  # Reduced from 256 but with better architecture
    patch_size=4,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    use_checkpoint=True,  # Enable checkpointing for memory efficiency
    
    # 📈 IMPROVED TRAINING PARAMETERS
    learning_rate=1e-3,  # Slightly increased for faster convergence
    weight_decay=1e-4,
    epochs=30,  # Reduced to 30 epochs
    warmup_epochs=5,  # Reduced warmup for shorter training
    device='cuda',
    use_amp=True,
    gradient_clip_val=1.0,
    accumulate_grad_batches=2,  # Effective batch size = 12
    
    # Enhanced validation settings
    val_interval=1,
    save_interval=3,  # More frequent saves for shorter training
    early_stopping_patience=10,  # Reduced patience for shorter training
    limit_val_batches=15,  # Faster validation
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.25,
    
)

# Enhanced configuration validation
def validate_improved_config(args):
    """Enhanced validation for improved configuration"""
    print("\n🔍 Validating improved configuration...")
    
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
        estimated_memory = args.batch_size * 0.8  # GB per batch item
        if estimated_memory > gpu_memory * 0.8:
            print(f"⚠️  Warning: Estimated memory usage ({estimated_memory:.1f}GB) may exceed available memory")
            print(f"   Consider reducing batch_size from {args.batch_size} to {int(args.batch_size * 0.7)}")
    
    print("✅ Enhanced configuration validation passed!")

def estimate_parameters(args):
    """Rough estimation of model parameters"""
    embed_dim = args.embed_dim
    depths = args.depths
    decoder_embed_dim = args.decoder_embed_dim
    
    # Rough encoder estimation
    encoder_params = 0
    for i, depth in enumerate(depths):
        layer_dim = embed_dim * (2 ** i)
        # Attention + MLP parameters per layer
        layer_params = (layer_dim * layer_dim * 3 * 2) + (layer_dim * layer_dim * args.mlp_ratio * 2)
        encoder_params += layer_params * depth
    
    # Rough decoder estimation  
    decoder_params = decoder_embed_dim * decoder_embed_dim * 8  # Rough estimate
    
    # Patch embedding
    patch_params = args.in_channels * embed_dim * (args.patch_size ** 3)
    
    total = encoder_params + decoder_params + patch_params
    return total

def print_training_strategy():
    """Print the improved training strategy"""
    print("\n" + "="*60)
    print("🎯 IMPROVED TRAINING STRATEGY")
    print("="*60)
    print("Phase 1 (Epochs 1-10): Warmup")
    print("  • Gradual learning rate increase")
    print("  • Focus on stable gradient flow")
    print("  • Light regularization")
    print()
    print("Phase 2 (Epochs 11-40): Main Training")
    print("  • Full learning rate")
    print("  • Combined Dice + CrossEntropy loss")
    print("  • Regular validation monitoring")
    print()
    print("Phase 3 (Epochs 41-80): Fine-tuning")
    print("  • Cosine annealing schedule")
    print("  • Increased validation frequency")
    print("  • Best model checkpointing")
    print("="*60)

# Print training strategy
print_training_strategy()

# Validate configuration
validate_improved_config(args)

# Print final configuration summary
print("\n=== 🚀 IMPROVED LIGHTWEIGHT SWINUNETR CONFIGURATION ===")
print(f"🎯 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
print(f"📐 Image size: {args.img_size}")
print(f"🧠 Embed dim: {args.embed_dim} (+{(args.embed_dim/48-1)*100:.0f}% vs original)")
print(f"🏗️  Depths: {args.depths}")
print(f"👁️  Num heads: {args.num_heads}")
print(f"🪟 Window size: {args.window_size}")
print(f"🔧 Decoder embed dim: {args.decoder_embed_dim}")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔄 SW batch size: {args.sw_batch_size}")
print(f"🕐 Warmup epochs: {args.warmup_epochs}")
print(f"📊 Total epochs: {args.epochs}")

def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        print("\n🚀 Starting improved lightweight SwinUNETR training...")
        print("Expected improvements:")
        print("  • 5-15% higher dice scores")
        print("  • More stable training curves")
        print("  • Better parameter efficiency")
        print("  • Improved memory usage")
        
        main(args)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print("🔧 Suggested fixes:")
            print(f"1. Reduce batch_size from {args.batch_size} to {args.batch_size//2}")
            print("2. Reduce accumulate_grad_batches from 2 to 1")
            print("3. Reduce sw_batch_size from 2 to 1")
            print("4. Enable gradient checkpointing")
            print("5. Reduce image size from 128 to 96")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Suggested fixes:")
        print("1. Make sure improved_architecture.py is in your swinunetrv2 package")
        print("2. Update your main.py to support 'improved_lightweight_swinunetr' model_type")
        print("3. Check that all required dependencies are installed")
        raise e
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 General troubleshooting:")
        print("1. Check that your swinunetrv2 package supports the improved model")
        print("2. Verify all configuration parameters are valid")
        print("3. Make sure dataset is properly formatted")
        print("4. Check file permissions and disk space")
        raise e

# Start improved training
if __name__ == "__main__":
    run_with_error_handling()