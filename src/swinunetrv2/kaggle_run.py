# run.py - Ultra-Efficient SwinUNETR with SegFormer3D-like efficiency
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

def estimate_ultra_efficient_parameters(efficiency_level, feature_size=None):
    """Estimate parameters for Ultra-Efficient SwinUNETR"""
    estimates = {
        "ultra": (5, 8),      # 5-8M parameters (SegFormer3D-like)
        "high": (10, 15),     # 10-15M parameters
        "balanced": (15, 25), # 15-25M parameters  
        "performance": (25, 35) # 25-35M parameters
    }
    
    if efficiency_level in estimates:
        return estimates[efficiency_level]
    elif feature_size:
        # Rough estimate based on feature size
        base_estimate = (feature_size / 4) ** 2
        return (base_estimate * 0.8, base_estimate * 1.2)
    else:
        return (15, 25)  # Default


# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Apply enhanced GPU optimizations
optimize_gpu_usage()

# 🎯 ULTRA-EFFICIENT SWINUNETR CONFIGURATION OPTIONS

def get_segformer_style_config():
    """SegFormer3D-style ultra-efficient - ~5-8M parameters"""
    return {
        "efficiency_level": "ultra",
        "use_segformer_style": True,
        "batch_size": 8,
        "accumulate_grad_batches": 2,  # effective batch = 16
        "learning_rate": 1e-3,  # Higher LR for smaller model
        "sw_batch_size": 4,  # Can afford larger sliding window batch
    }

def get_ultra_lightweight_config():
    """Ultra lightweight - ~8-12M parameters"""
    return {
        "efficiency_level": "ultra",
        "feature_size": 16,
        "depths": (1, 1, 1, 1),
        "num_heads": (1, 2, 4, 8),
        "decoder_channels": (64, 32, 16, 8),
        "batch_size": 6,
        "accumulate_grad_batches": 2,  # effective batch = 12
        "learning_rate": 8e-4,
    }

def get_high_efficiency_config():
    """High efficiency - ~12-18M parameters"""
    return {
        "efficiency_level": "high",
        "feature_size": 20,
        "depths": (1, 1, 2, 1),
        "num_heads": (2, 4, 6, 12),
        "decoder_channels": (80, 40, 20, 10),
        "batch_size": 5,
        "accumulate_grad_batches": 3,  # effective batch = 15
        "learning_rate": 6e-4,
    }

def get_balanced_efficiency_config():
    """Balanced efficiency - ~18-25M parameters"""
    return {
        "efficiency_level": "balanced",
        "feature_size": 24,
        "depths": (1, 1, 2, 1),
        "num_heads": (2, 4, 8, 16),
        "decoder_channels": (96, 48, 24, 12),
        "batch_size": 4,
        "accumulate_grad_batches": 3,  # effective batch = 12
        "learning_rate": 5e-4,
    }

def get_performance_efficiency_config():
    """Performance efficiency - ~25-35M parameters"""
    return {
        "efficiency_level": "performance",
        "feature_size": 32,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "decoder_channels": (128, 64, 32, 16),
        "batch_size": 3,
        "accumulate_grad_batches": 4,  # effective batch = 12
        "learning_rate": 3e-4,
    }

# 🚀 CHOOSE YOUR CONFIGURATION
# Change this to experiment with different efficiency levels
config_type = "segformer_style"  # Options: "segformer_style", "ultra_lightweight", "high_efficiency", "balanced_efficiency", "performance_efficiency"

config_map = {
    "segformer_style": get_segformer_style_config(),
    "ultra_lightweight": get_ultra_lightweight_config(),
    "high_efficiency": get_high_efficiency_config(),
    "balanced_efficiency": get_balanced_efficiency_config(),
    "performance_efficiency": get_performance_efficiency_config(),
}

if config_type in config_map:
    model_config = config_map[config_type]
    param_range = estimate_ultra_efficient_parameters(
        model_config.get("efficiency_level", "balanced"),
        model_config.get("feature_size")
    )
    print(f"🚀 Using {config_type.upper().replace('_', ' ')} configuration")
    print(f"📊 Estimated parameters: {param_range[0]}-{param_range[1]}M")
else:
    print(f"❌ Unknown configuration: {config_type}")
    print("Available options: segformer_style, ultra_lightweight, high_efficiency, balanced_efficiency, performance_efficiency")
    exit(1)

# 🔧 MAIN CONFIGURATION WITH ULTRA-EFFICIENT SWINUNETR
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=model_config["batch_size"],
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    
    # 🚀 ULTRA-EFFICIENT SWINUNETR PARAMETERS
    img_size=128,
    in_channels=4,
    out_channels=3,
    
    # Efficiency configuration
    efficiency_level=model_config.get("efficiency_level", "balanced"),
    use_segformer_style=model_config.get("use_segformer_style", False),
    
    # Model architecture (will override efficiency_level if specified)
    feature_size=model_config.get("feature_size", 24),
    depths=model_config.get("depths", (1, 1, 2, 1)),
    num_heads=model_config.get("num_heads", (2, 4, 8, 16)),
    decoder_channels=model_config.get("decoder_channels", (96, 48, 24, 12)),
    
    # Standard parameters
    drop_rate=0.1,
    attn_drop_rate=0.1,
    dropout_path_rate=0.1,
    use_checkpoint=True,
    use_v2=True,
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
    early_stopping_patience=15,
    limit_val_batches=5,
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=model_config.get("sw_batch_size", 2),
    overlap=0.25,
    
    # Legacy parameters (will be ignored)
    embed_dim=None,
    window_size=None,
    mlp_ratio=None,
    decoder_embed_dim=None,
    patch_size=None,
)

def validate_ultra_efficient_config(args):
    """Validate Ultra-Efficient SwinUNETR configuration"""
    print("\n🔍 Validating Ultra-Efficient SwinUNETR configuration...")
    
    # Basic validations
    if hasattr(args, 'depths') and hasattr(args, 'num_heads'):
        assert len(args.depths) == len(args.num_heads), "depths and num_heads must have same length"
    assert args.feature_size > 0, "feature_size must be positive"
    assert args.batch_size > 0, "batch_size must be positive"
    
    # Parameter estimation
    param_range = estimate_ultra_efficient_parameters(args.efficiency_level, args.feature_size)
    print(f"📊 Estimated parameters: {param_range[0]}-{param_range[1]}M")
    
    # Efficiency comparison
    efficiency_configs = {
        "SegFormer3D": "8-15M",
        "Standard SwinUNETR": "62M",
        "UNet3D": "30-40M", 
        "nnU-Net": "31M"
    }
    
    print(f"📋 Efficiency comparison:")
    for model, params in efficiency_configs.items():
        print(f"   • {model}: {params}")
    
    # Memory check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔧 Available GPU memory: {gpu_memory:.1f} GB")
        
        # Memory estimation for Ultra-Efficient SwinUNETR
        estimated_params = (param_range[0] + param_range[1]) / 2 * 1e6
        base_memory = 1.5  # Lower base memory for efficient model
        memory_per_sample = 0.4 if args.feature_size <= 24 else 0.6  # Much lower memory per sample
        estimated_memory = base_memory + (args.batch_size * memory_per_sample)
        
        print(f"💾 Estimated memory usage: {estimated_memory:.1f} GB")
        
        if estimated_memory > gpu_memory * 0.85:
            print(f"⚠️  Warning: May exceed GPU memory!")
            new_batch_size = int((gpu_memory * 0.8 - base_memory) / memory_per_sample)
            print(f"   Suggested batch_size: {new_batch_size}")
        else:
            print(f"✅ Memory usage looks good!")
    
    print("✅ Ultra-Efficient SwinUNETR configuration validated!")

# Validate configuration
validate_ultra_efficient_config(args)

# Print detailed configuration summary
print(f"\n=== 🚀 ULTRA-EFFICIENT SWINUNETR CONFIGURATION ===")
print(f"🎯 Configuration: {config_type.upper().replace('_', ' ')}")
print(f"⚡ Efficiency level: {args.efficiency_level}")
if args.use_segformer_style:
    print(f"🔥 SegFormer3D-style: {args.use_segformer_style}")
print(f"🏗️  Feature size: {args.feature_size}")
print(f"📐 Depths: {args.depths}")
print(f"👁️  Num heads: {args.num_heads}")
print(f"🔧 Decoder channels: {args.decoder_channels}")
print(f"📦 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔄 Use checkpoint: {args.use_checkpoint}")
print(f"✨ SwinUNETR-V2: {args.use_v2}")
print(f"📏 Norm: {args.norm_name}")

def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        param_range = estimate_ultra_efficient_parameters(args.efficiency_level, args.feature_size)
        print(f"\n🚀 Starting Ultra-Efficient SwinUNETR training ({config_type})...")
        print("Expected benefits:")
        print(f"  • SegFormer3D-like efficiency: {param_range[0]}-{param_range[1]}M parameters")
        print("  • Lightweight decoder architecture")
        print("  • Optimized attention mechanisms")
        print("  • Proven MONAI backbone")
        print("  • Better memory efficiency")
        print("  • Faster training and inference")
        
        main(args)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print("🔧 Try these fixes in order:")
            print(f"1. Switch to 'segformer_style' or 'ultra_lightweight' config")
            print(f"2. Reduce batch_size from {args.batch_size} to {args.batch_size//2}")
            print("3. Reduce accumulate_grad_batches")
            print("4. Set sw_batch_size to 1")
            print("5. The model should already be very memory efficient!")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure you have:")
        print("1. Updated architecture.py with Ultra-Efficient SwinUNETR")
        print("2. Updated pipeline.py with new model integration")
        print("3. MONAI installed: pip install monai")
        print("4. All dependencies: torch, pytorch-lightning")
        raise e
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that architecture.py contains UltraEfficientSwinUNETR")
        print("2. Verify pipeline.py imports from architecture")
        print("3. Check MONAI version: pip show monai")
        print("4. Verify dataset format")
        raise e

# Configuration switching helper
def switch_config(new_config):
    """Helper to switch between configurations"""
    if new_config in config_map:
        print(f"\n🔄 Switching to {new_config} configuration...")
        config = config_map[new_config]
        for key, value in config.items():
            setattr(args, key, value)
        validate_ultra_efficient_config(args)
        return True
    else:
        print(f"❌ Unknown configuration: {new_config}")
        print(f"Available: {list(config_map.keys())}")
        return False

def compare_all_configs():
    """Compare all available efficiency configurations"""
    print("\n📊 ALL ULTRA-EFFICIENT CONFIGURATIONS:")
    print("=" * 60)
    
    for config_name, config in config_map.items():
        param_range = estimate_ultra_efficient_parameters(
            config.get("efficiency_level", "balanced"),
            config.get("feature_size")
        )
        effective_batch = config["batch_size"] * config["accumulate_grad_batches"]
        
        print(f"\n🎯 {config_name.upper().replace('_', ' ')}:")
        print(f"   Parameters: {param_range[0]}-{param_range[1]}M")
        print(f"   Batch size: {config['batch_size']} (effective: {effective_batch})")
        print(f"   Learning rate: {config['learning_rate']}")
        if 'feature_size' in config:
            print(f"   Feature size: {config['feature_size']}")
        if config.get('use_segformer_style'):
            print(f"   SegFormer3D-style: Yes")

# Start training
if __name__ == "__main__":
    print(f"\n💡 Current configuration: {config_type}")
    print(f"💡 To switch configurations, change 'config_type' at the top of this file")
    print(f"💡 To see all options, uncomment compare_all_configs() below")
    
    # Uncomment to see all configuration options
    # compare_all_configs()
    
    run_with_error_handling()