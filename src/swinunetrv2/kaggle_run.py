# run.py
from swinunetrv2.kaggle_setup import setup_kaggle_notebook
from swinunetrv2.main import main
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

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

def estimate_monai_parameters(feature_size, depths):
    """Estimate MONAI SwinUNETR parameters"""
    # More accurate estimation for MONAI SwinUNETR
    
    # Base parameters (rough estimates from MONAI source)
    patch_embed_params = feature_size * 4 * 2 * 2 * 2  # patch embedding
    
    # Encoder parameters
    encoder_params = 0
    for i, depth in enumerate(depths):
        layer_dim = feature_size * (2 ** i) if i > 0 else feature_size
        # Swin blocks: attention + MLP
        encoder_params += depth * (layer_dim * layer_dim * 3 + layer_dim * layer_dim * 4)
    
    # Decoder parameters (SwinUNETR decoder)
    decoder_params = feature_size * 3 * 16  # upsampling layers
    
    total_params = patch_embed_params + encoder_params + decoder_params
    return int(total_params)


# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Apply enhanced GPU optimizations
optimize_gpu_usage()

# 🎯 MONAI SWINUNETR CONFIGURATION OPTIONS

def get_ultra_lightweight_config():
    """Ultra lightweight MONAI SwinUNETR - ~15M parameters"""
    return {
        "feature_size": 24,
        "depths": (2, 2, 2, 2),
        "num_heads": (2, 4, 8, 16),
        "batch_size": 6,
        "accumulate_grad_batches": 2,  # effective batch = 12
        "learning_rate": 8e-4,
    }

def get_balanced_config():
    """Balanced MONAI SwinUNETR - ~28M parameters"""
    return {
        "feature_size": 32,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "batch_size": 4,
        "accumulate_grad_batches": 3,  # effective batch = 12
        "learning_rate": 5e-4,
    }

def get_performance_config():
    """Performance MONAI SwinUNETR - ~62M parameters (original)"""
    return {
        "feature_size": 48,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "batch_size": 2,
        "accumulate_grad_batches": 6,  # effective batch = 12
        "learning_rate": 1e-4,
    }

# Choose configuration (change this to experiment)
config_type = "balanced"  # Options: "ultra_lightweight", "balanced", "performance"

if config_type == "ultra_lightweight":
    model_config = get_ultra_lightweight_config()
    print("🚀 Using Ultra Lightweight MONAI SwinUNETR (~15M params)")
elif config_type == "balanced":
    model_config = get_balanced_config()
    print("🚀 Using Balanced MONAI SwinUNETR (~28M params)")
else:
    model_config = get_performance_config()
    print("🚀 Using Performance MONAI SwinUNETR (~62M params)")

# 🔧 MAIN CONFIGURATION WITH MONAI SWINUNETR
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=model_config["batch_size"],
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    
    # 🚀 MONAI SWINUNETR PARAMETERS
    img_size=128,
    in_channels=4,
    out_channels=3,
    feature_size=model_config["feature_size"],  # Key parameter for model size
    depths=model_config["depths"],              # Transformer depths
    num_heads=model_config["num_heads"],        # Attention heads
    drop_rate=0.1,
    attn_drop_rate=0.1,
    dropout_path_rate=0.1,
    use_checkpoint=True,
    use_v2=True,                               # Enable SwinUNETR-V2
    norm_name="instance",
    
    # Training parameters
    learning_rate=model_config["learning_rate"],
    weight_decay=1e-5,
    epochs=30,
    warmup_epochs=5,
    device='cuda',
    use_amp=True,
    gradient_clip_val=1.0,
    accumulate_grad_batches=model_config["accumulate_grad_batches"],
    
    # Validation settings
    val_interval=1,
    save_interval=3,
    early_stopping_patience=15,  # More patience for MONAI model
    limit_val_batches=5,
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.25,
    
    # Legacy parameters (will be ignored but kept for compatibility)
    embed_dim=None,
    window_size=None,
    mlp_ratio=None,
    decoder_embed_dim=None,
    patch_size=None,
)

def validate_monai_config(args):
    """Validate MONAI SwinUNETR configuration"""
    print("\n🔍 Validating MONAI SwinUNETR configuration...")
    
    # Basic validations
    assert len(args.depths) == len(args.num_heads), "depths and num_heads must have same length"
    assert args.feature_size > 0, "feature_size must be positive"
    assert args.batch_size > 0, "batch_size must be positive"
    
    # Parameter estimation
    estimated_params = estimate_monai_parameters(args.feature_size, args.depths)
    print(f"📊 Estimated parameters: {estimated_params/1e6:.1f}M")
    
    # Memory check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔧 Available GPU memory: {gpu_memory:.1f} GB")
        
        # Memory estimation for MONAI SwinUNETR
        base_memory = 2.0  # Base memory for model
        memory_per_sample = 0.8 if args.feature_size <= 32 else 1.2  # Memory per sample
        estimated_memory = base_memory + (args.batch_size * memory_per_sample)
        
        print(f"💾 Estimated memory usage: {estimated_memory:.1f} GB")
        
        if estimated_memory > gpu_memory * 0.85:
            print(f"⚠️  Warning: May exceed GPU memory!")
            new_batch_size = int((gpu_memory * 0.8 - base_memory) / memory_per_sample)
            print(f"   Suggested batch_size: {new_batch_size}")
    
    print("✅ MONAI SwinUNETR configuration validated!")

# Validate configuration
validate_monai_config(args)

# Print configuration summary
print(f"\n=== 🚀 MONAI SWINUNETR-V2 CONFIGURATION ({config_type.upper()}) ===")
print(f"🎯 Feature size: {args.feature_size} (main parameter control)")
print(f"🏗️  Depths: {args.depths}")
print(f"👁️  Num heads: {args.num_heads}")
print(f"📦 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔄 Use checkpoint: {args.use_checkpoint}")
print(f"✨ SwinUNETR-V2: {args.use_v2}")
print(f"📏 Norm: {args.norm_name}")
print(f"🎲 Dropout rates: {args.drop_rate}/{args.attn_drop_rate}/{args.dropout_path_rate}")

def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        print(f"\n🚀 Starting MONAI SwinUNETR-V2 training ({config_type})...")
        print("Expected benefits:")
        print("  • Proven architecture from MONAI")
        print("  • Proper weight initialization")
        print("  • Optimized transformer blocks")
        print("  • SwinUNETR-V2 improvements")
        print("  • Better TC/WT/ET segmentation")
        
        main(args)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print("🔧 Try these fixes in order:")
            print(f"1. Switch to 'ultra_lightweight' config (change config_type)")
            print(f"2. Reduce batch_size from {args.batch_size} to {args.batch_size//2}")
            print("3. Reduce accumulate_grad_batches")
            print("4. Set use_checkpoint=True (already enabled)")
            print("5. Reduce sw_batch_size from 2 to 1")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure you have:")
        print("1. MONAI installed: pip install monai")
        print("2. Updated pipeline.py with MONAI SwinUNETR")
        print("3. All dependencies: torch, pytorch-lightning")
        raise e
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check MONAI version: pip show monai")
        print("2. Verify dataset format")
        print("3. Check file permissions")
        raise e

# Configuration switching helper
def switch_config(new_config):
    """Helper to switch between configurations"""
    configs = {
        "ultra_lightweight": get_ultra_lightweight_config(),
        "balanced": get_balanced_config(),
        "performance": get_performance_config()
    }
    
    if new_config in configs:
        print(f"\n🔄 Switching to {new_config} configuration...")
        config = configs[new_config]
        for key, value in config.items():
            setattr(args, key, value)
        validate_monai_config(args)
        return True
    else:
        print(f"❌ Unknown configuration: {new_config}")
        return False

# Start training
if __name__ == "__main__":
    print(f"\n💡 To switch configurations, change 'config_type' at the top of this file")
    print(f"   Current: {config_type}")
    print(f"   Options: ultra_lightweight, balanced, performance")
    run_with_error_handling()