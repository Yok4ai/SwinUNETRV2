# run_hybrid.py - Simple script to run Hybrid SwinUNETR-SegFormer3D
from swinunetrv2.kaggle_setup import setup_kaggle_notebook
from swinunetrv2.main import main as train_model
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")


def create_hybrid_args(preset="light"):
    """Create arguments for Hybrid SwinUNETR-SegFormer3D with memory optimization"""
    
    # Get preset configuration
    from swinunetrv2.models.pipeline import get_hybrid_config
    config = get_hybrid_config(preset)
    
    return argparse.Namespace(
        # Data parameters
        input_dir='/kaggle/working',
        batch_size=config["batch_size"],
        num_workers=2,  # Reduced from 4
        pin_memory=True,
        
        # Model parameters - HYBRID SPECIFIC
        img_size=96,  # Reduced from 128
        in_channels=4,
        out_channels=3,
        
        # Hybrid configuration
        efficiency_level=config["efficiency_level"],
        use_segformer_decoder=True,
        decoder_embedding_dim=config["decoder_embedding_dim"],
        
        # SwinUNETR backbone parameters
        feature_size=24,  # Reduced default
        depths=(1, 1, 1, 1),  # Reduced default
        num_heads=(2, 4, 8, 16),
        
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
        early_stopping_patience=15,
        
        # Inference parameters
        roi_size=[96, 96, 96],  # Reduced from 128
        sw_batch_size=1,  # Reduced from 2
        overlap=0.15,  # Reduced from 0.25
        
        # Architecture parameters
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        decoder_dropout=0.0,
        use_checkpoint=True,
        use_v2=True,
        norm_name="instance",
    )


def run_hybrid_training():
    """Run Hybrid SwinUNETR-SegFormer3D training"""
    
    # Setup environment
    output_dir = setup_kaggle_notebook()
    print(f"Dataset ready: {output_dir}")
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        print("🔥 GPU optimized!")
    
    # Create hybrid configuration
    args = create_hybrid_args(preset="light")  # Changed from "balanced" to "light"
    
    print(f"\n🚀 HYBRID SWINUNETR-SEGFORMER3D")
    print(f"🏗️  Backbone: SwinUNETR V2 (feature_size will be set by efficiency_level)")
    print(f"🎯 Decoder: SegFormer3D-style MLP decoder")
    print(f"📊 Efficiency level: {args.efficiency_level}")
    print(f"🔧 Decoder embedding dim: {args.decoder_embedding_dim}")
    print(f"📦 Batch size: {args.batch_size} (effective: {args.batch_size * args.accumulate_grad_batches})")
    print(f"⚡ Learning rate: {args.learning_rate}")
    print(f"🔄 V2 merging: {args.use_v2}")
    
    # Run training
    try:
        train_model(args)
        print("✅ HYBRID TRAINING SUCCESSFUL!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        if "out of memory" in str(e).lower():
            print("\n🔧 Memory optimization suggestions:")
            print("1. Switch to preset='light' for smaller model")
            print("2. Reduce batch_size to 2")
            print("3. Reduce sw_batch_size to 1")
            print("4. The hybrid model should be more efficient than standard SwinUNETR")
        
        raise e


def run_comparison_training():
    """Run both standard and hybrid for comparison"""
    
    print("🏃‍♂️ Running comparison: Standard vs Hybrid SwinUNETR")
    
    # First run standard (your existing fixed version)
    print("\n" + "="*50)
    print("🔴 TRAINING STANDARD ULTRA-EFFICIENT SWINUNETR")
    print("="*50)
    
    # Import your existing run function
    from .run import run_training as run_standard
    try:
        run_standard()
        print("✅ Standard training completed")
    except Exception as e:
        print(f"❌ Standard training failed: {e}")
    
    # Then run hybrid
    print("\n" + "="*50)
    print("🟢 TRAINING HYBRID SWINUNETR-SEGFORMER3D")
    print("="*50)
    
    try:
        run_hybrid_training()
        print("✅ Hybrid training completed")
    except Exception as e:
        print(f"❌ Hybrid training failed: {e}")
    
    print("\n🏁 Comparison completed!")


if __name__ == "__main__":
    # Choose what to run:
    
    # Option 1: Run just the hybrid model
    run_hybrid_training()
    
    # Option 2: Run comparison (uncomment to use)
    # run_comparison_training()
    
    # Option 3: Run with different preset (uncomment to use)
    # For lighter model: change create_hybrid_args(preset="light")
    # For performance model: change create_hybrid_args(preset="performance")