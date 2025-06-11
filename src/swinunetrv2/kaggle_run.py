# Import necessary modules
from swinunetrv2.kaggle_setup import setup_kaggle_notebook
from swinunetrv2.main import main
import argparse
import torch

# 🔥 ADDITIONAL GPU OPTIMIZATION SETTINGS
def optimize_gpu_usage():
    """Apply additional GPU optimizations"""
    
    # Enable cudNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable TensorFloat-32 for better performance on Ampere GPUs
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set memory management
    torch.cuda.empty_cache()
    
    print("🔥 GPU optimizations applied!")

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Apply GPU optimizations
optimize_gpu_usage()

# Configure training parameters
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,  # Disabled to fix pickling error
    
    # Model parameters
    img_size=128,
    in_channels=4,  # 4 modalities for BraTS (t1c, t1n, t2f, t2w)
    out_channels=3,  # 3 tumor regions for BraTS
    feature_size=48,
    embed_dim=48,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    decoder_embed_dim=256,
    patch_size=4,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    use_checkpoint=True,  # Enable checkpointing for memory efficiency
    
    # Training parameters
    learning_rate=2e-3,
    weight_decay=1e-4,
    epochs=50,
    warmup_epochs=5,
    device='cuda',
    use_amp=True,
    gradient_clip_val=1.0,
    
    # Validation settings
    val_interval=1,
    save_interval=10,
    early_stopping_patience=15,
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.25
)

# Print model configuration
print("=== Lightweight SwinUNETR Configuration ===")
print(f"🎯 Batch size: {args.batch_size}")
print(f"📐 Image size: {args.img_size}")
print(f"🧠 Embed dim: {args.embed_dim}")
print(f"🏗️  Depths: {args.depths}")
print(f"👁️  Num heads: {args.num_heads}")
print(f"🔧 Decoder embed dim: {args.decoder_embed_dim}")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔄 SW batch size: {args.sw_batch_size}")

# Validate configuration
def validate_config(args):
    """Validate that the configuration makes sense"""
    assert args.embed_dim == args.feature_size, f"embed_dim ({args.embed_dim}) should match feature_size ({args.feature_size})"
    assert len(args.depths) == len(args.num_heads), "depths and num_heads must have same length"
    assert args.window_size <= args.img_size // 4, f"window_size ({args.window_size}) too large for img_size ({args.img_size})"
    assert args.patch_size <= args.img_size // 8, f"patch_size ({args.patch_size}) too large for img_size ({args.img_size})"
    print("✅ Configuration validation passed!")

# Validate before training
validate_config(args)

# Start training
try:
    main(args)
except Exception as e:
    print(f"❌ Training failed with error: {e}")
    print("\n🔧 Troubleshooting suggestions:")
    print("1. Check if your swinunetrv2 package supports the lightweight_swinunetr model_type")
    print("2. Verify that all required parameters are present in your main.py")
    print("3. Make sure the lightweight model code has been integrated into your architecture.py")
    print("4. Check GPU memory if you get CUDA out of memory errors")
    raise e