# run.py - Ultra-Efficient SwinUNETR with SegFormer3D-like efficiency
from swinunetrv2.kaggle_setup import setup_kaggle_notebook
from swinunetrv2.main import main
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

def optimize_gpu_usage():
    """Apply enhanced GPU optimizations for better performance"""
    if not torch.cuda.is_available():
        return
        
    # Enable cudNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable TensorFloat-32 for better performance on Ampere GPUs
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set memory management
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass
    
    print("🔥 Enhanced GPU optimizations applied!")

def get_efficiency_config(efficiency_level="balanced"):
    """Get configuration for specified efficiency level"""
    configs = {
        "ultra": {  # SegFormer3D-like efficiency (5-8M params)
            "feature_size": 12,
            "depths": (1, 1, 1, 1),
            "num_heads": (1, 2, 4, 8),
            "decoder_channels": (48, 24, 12, 6),
            "batch_size": 8,
            "accumulate_grad_batches": 2,
            "learning_rate": 1e-3,
            "sw_batch_size": 4,
        },
        "high": {  # High efficiency (10-15M params)
            "feature_size": 24,
            "depths": (1, 1, 2, 1),
            "num_heads": (2, 4, 6, 12),
            "decoder_channels": (96, 48, 24, 12),
            "batch_size": 5,
            "accumulate_grad_batches": 3,
            "learning_rate": 6e-4,
        },
        "balanced": {  # Balanced efficiency (15-25M params)
            "feature_size": 24,
            "depths": (1, 1, 2, 1),
            "num_heads": (2, 4, 8, 16),
            "decoder_channels": (96, 48, 24, 12),
            "batch_size": 4,
            "accumulate_grad_batches": 3,
            "learning_rate": 5e-4,
        },
        "performance": {  # Performance focused (25-35M params)
            "feature_size": 36,
            "depths": (2, 2, 2, 2),
            "num_heads": (3, 6, 12, 24),
            "decoder_channels": (144, 72, 36, 18),
            "batch_size": 3,
            "accumulate_grad_batches": 4,
            "learning_rate": 3e-4,
        }
    }
    
    if efficiency_level not in configs:
        raise ValueError(f"Unknown efficiency level: {efficiency_level}")
    
    return configs[efficiency_level]

def create_args(efficiency_level="balanced", use_segformer_style=False):
    """Create argument namespace with specified efficiency configuration"""
    config = get_efficiency_config(efficiency_level)
    
    return argparse.Namespace(
        # Data parameters
        input_dir='/kaggle/working',
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        
        # Model parameters
        img_size=128,
        in_channels=4,
        out_channels=3,
        
        # Efficiency configuration
        efficiency_level=efficiency_level,
        use_segformer_style=use_segformer_style,
        feature_size=config["feature_size"],
        depths=config["depths"],
        num_heads=config["num_heads"],
        decoder_channels=config["decoder_channels"],
        
        # Training parameters
        learning_rate=config["learning_rate"],
        weight_decay=1e-5,
        epochs=30,
        warmup_epochs=5,
        device='cuda',
        use_amp=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        
        # Validation settings
        val_interval=1,
        save_interval=3,
        early_stopping_patience=15,
        limit_val_batches=5,
        
        # Inference parameters
        roi_size=[128, 128, 128],
        sw_batch_size=config.get("sw_batch_size", 2),
        overlap=0.25,
        
        # Model architecture
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        use_checkpoint=True,
        use_v2=True,
        norm_name="instance",
    )

def validate_config(args):
    """Validate configuration parameters"""
    print("\n🔍 Validating configuration...")
    
    # Basic validations
    if hasattr(args, 'depths') and hasattr(args, 'num_heads'):
        assert len(args.depths) == len(args.num_heads), "depths and num_heads must have same length"
    assert args.feature_size > 0, "feature_size must be positive"
    assert args.batch_size > 0, "batch_size must be positive"
    
    # Print configuration summary
    print(f"\n=== 🚀 ULTRA-EFFICIENT SWINUNETR CONFIGURATION ===")
    print(f"🎯 Efficiency level: {args.efficiency_level}")
    if args.use_segformer_style:
        print(f"🔥 SegFormer3D-style: Enabled")
    print(f"🏗️  Feature size: {args.feature_size}")
    print(f"📐 Depths: {args.depths}")
    print(f"👁️  Num heads: {args.num_heads}")
    print(f"🔧 Decoder channels: {args.decoder_channels}")
    print(f"📦 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
    print(f"⚡ Learning rate: {args.learning_rate}")
    
    # Memory check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🔧 Available GPU memory: {gpu_memory:.1f} GB")
        
        # Rough memory estimation
        estimated_memory = 1.5 + (args.batch_size * 0.4)
        print(f"💾 Estimated memory usage: {estimated_memory:.1f} GB")
        
        if estimated_memory > gpu_memory * 0.85:
            print(f"⚠️  Warning: May exceed GPU memory!")
            new_batch_size = int((gpu_memory * 0.8 - 1.5) / 0.4)
            print(f"   Suggested batch_size: {new_batch_size}")
    
    print("\n✅ Configuration validated!")

def main():
    """Main training execution"""
    # Setup environment
    output_dir = setup_kaggle_notebook()
    print(f"Dataset prepared in: {output_dir}")
    
    # Apply GPU optimizations
    optimize_gpu_usage()
    
    # Create and validate configuration
    args = create_args(efficiency_level="balanced", use_segformer_style=False)
    validate_config(args)
    
    # Run training
    try:
        main(args)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if "out of memory" in str(e).lower():
            print("\n🔧 Try these fixes:")
            print("1. Switch to 'ultra' efficiency level")
            print("2. Reduce batch_size")
            print("3. Reduce sw_batch_size to 1")
        raise e

if __name__ == "__main__":
    main()