# trainer.py - Updated for Ultra-Efficient SwinUNETR
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from .pipeline import BrainTumorSegmentation


def setup_training(train_ds, val_ds, args):
    """Setup training with Ultra-Efficient SwinUNETR configurations"""
    
    # Validate feature size is divisible by 12
    if args.feature_size % 12 != 0:
        raise ValueError(f"feature_size must be divisible by 12, got {args.feature_size}")
    
    # Enhanced data loaders with efficiency optimizations
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )

    # Enhanced early stopping for efficient models
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.001,
        patience=args.early_stopping_patience,
        verbose=True,
        mode='max',
        check_finite=True
    )
    
    # Model checkpointing with efficiency-aware naming
    efficiency_level = getattr(args, 'efficiency_level', 'custom')
    use_segformer_style = getattr(args, 'use_segformer_style', False)
    
    checkpoint_name = "ultra-efficient-swinunetr"
    if use_segformer_style:
        checkpoint_name += "-segformer-style"
    else:
        checkpoint_name += f"-{efficiency_level}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename=f"{checkpoint_name}-{{epoch:02d}}-{{val_mean_dice:.4f}}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False
    )
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Timer callback
    timer_callback = Timer(duration="00:12:00:00")  # 12 hours

    # Initialize wandb with Ultra-Efficient SwinUNETR tracking
    wandb_project_name = "brain-tumor-ultra-efficient-swinunetr"
    
    # Create detailed wandb run name
    if use_segformer_style:
        run_name = f"segformer-style-{args.feature_size}feat"
    else:
        run_name = f"{efficiency_level}-{args.feature_size}feat"
    
    wandb_config = {
        "architecture": "Ultra_Efficient_SwinUNETR",
        "efficiency_level": efficiency_level,
        "use_segformer_style": use_segformer_style,
        "feature_size": args.feature_size,
        "depths": args.depths,
        "num_heads": args.num_heads,
        "decoder_channels": getattr(args, 'decoder_channels', 'N/A'),
        "use_checkpoint": args.use_checkpoint,
        "use_v2": args.use_v2,
        "norm_name": args.norm_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
        "epochs": args.epochs,
        "drop_rate": args.drop_rate,
        "attn_drop_rate": args.attn_drop_rate,
        "dropout_path_rate": args.dropout_path_rate,
        "sw_batch_size": args.sw_batch_size,
        "roi_size": args.roi_size
    }
    
    wandb.init(
        project=wandb_project_name, 
        name=run_name,
        config=wandb_config
    )
    wandb_logger = WandbLogger()

    # Initialize Ultra-Efficient SwinUNETR model
    model = BrainTumorSegmentation(
        train_loader,
        val_loader,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        # Ultra-Efficient SwinUNETR parameters
        efficiency_level=efficiency_level,
        use_segformer_style=use_segformer_style,
        feature_size=args.feature_size,
        depths=args.depths,
        num_heads=args.num_heads,
        decoder_channels=getattr(args, 'decoder_channels', (96, 48, 24, 12)),
        # Standard training parameters
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.dropout_path_rate,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        use_mixed_precision=args.use_amp,
        norm_name=args.norm_name,
        use_checkpoint=args.use_checkpoint,
        use_v2=args.use_v2,
        # Legacy parameters (will be ignored but kept for compatibility)
        img_size=getattr(args, 'img_size', 128),
        embed_dim=getattr(args, 'embed_dim', None),
        window_size=getattr(args, 'window_size', None),
        mlp_ratio=getattr(args, 'mlp_ratio', None),
        decoder_embed_dim=getattr(args, 'decoder_embed_dim', None),
        patch_size=getattr(args, 'patch_size', None)
    )

    # Setup trainer with efficiency optimizations
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if args.use_amp else '32',
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=5,
        callbacks=[
            early_stop_callback, 
            checkpoint_callback, 
            lr_monitor, 
            timer_callback
        ],
        limit_val_batches=args.limit_val_batches if hasattr(args, 'limit_val_batches') else None,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=False,
        benchmark=True,  # Optimize for consistent operations
        enable_checkpointing=True,
        enable_model_summary=True,
        sync_batchnorm=False,  # Not needed for single GPU
        enable_progress_bar=True,
        # Efficiency optimizations
        detect_anomaly=False,  # Disable for performance
        profiler=None,  # Disable profiling for efficiency
    )

    return model, trainer, train_loader, val_loader


def train_model(model, trainer, train_loader, val_loader):
    """Train the Ultra-Efficient SwinUNETR model with enhanced monitoring"""
    print("🚀 Starting Ultra-Efficient SwinUNETR training...")
    
    # Get efficiency summary
    efficiency_summary = model.get_efficiency_summary()
    total_params = efficiency_summary["total_parameters"]
    
    print(f"📊 Training {efficiency_summary['model_type']} with {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Log detailed model configuration
    print(f"🔧 Model Configuration:")
    print(f"   Efficiency level: {efficiency_summary['efficiency_level']}")
    print(f"   SegFormer-style: {efficiency_summary['segformer_style']}")
    print(f"   Feature size: {efficiency_summary['feature_size']}")
    print(f"   Depths: {efficiency_summary['depths']}")
    print(f"   Num heads: {efficiency_summary['num_heads']}")
    print(f"   Decoder channels: {efficiency_summary['decoder_channels']}")
    print(f"   Use V2: {model.hparams.use_v2}")
    print(f"   Use checkpoint: {model.hparams.use_checkpoint}")
    print(f"   Norm: {model.hparams.norm_name}")
    print(f"   Dropout rates: {model.hparams.drop_rate}/{model.hparams.attn_drop_rate}/{model.hparams.dropout_path_rate}")
    
    # Efficiency comparison
    print(f"\n📈 Efficiency Metrics:")
    print(f"   Parameter reduction: {efficiency_summary['parameter_reduction_vs_standard']}")
    print(f"   Estimated memory: {efficiency_summary['estimated_memory_gb']} GB")
    
    # Start training
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Final results with efficiency focus
        print(f"\n✅ Ultra-Efficient SwinUNETR training completed successfully!")
        print(f"📊 Results:")
        print(f"   Best validation Dice: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}")
        print(f"   Model efficiency: {efficiency_summary['efficiency_level']} level")
        print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        if len(model.metric_values_tc) > 0:
            print(f"   Final individual metrics:")
            print(f"     TC (Tumor Core): {model.metric_values_tc[-1]:.4f}")
            print(f"     WT (Whole Tumor): {model.metric_values_wt[-1]:.4f}")
            print(f"     ET (Enhancing Tumor): {model.metric_values_et[-1]:.4f}")
        
        # Save final model with efficiency-aware naming
        efficiency_level = efficiency_summary['efficiency_level']
        if efficiency_summary['segformer_style']:
            model_name = f"final_segformer_style_swinunetr_feat{efficiency_summary['feature_size']}.pth"
        else:
            model_name = f"final_ultra_efficient_swinunetr_{efficiency_level}_feat{efficiency_summary['feature_size']}.pth"
        
        torch.save(model.model.state_dict(), model_name)
        print(f"💾 Final model saved as: {model_name}")
        
        # Enhanced wandb logging
        if wandb.run is not None:
            wandb.log({
                "final_best_dice": model.best_metric,
                "final_tc": model.metric_values_tc[-1] if len(model.metric_values_tc) > 0 else 0,
                "final_wt": model.metric_values_wt[-1] if len(model.metric_values_wt) > 0 else 0,
                "final_et": model.metric_values_et[-1] if len(model.metric_values_et) > 0 else 0,
                "total_parameters": total_params,
                "efficiency_level": efficiency_level,
                "parameter_reduction_percent": float(efficiency_summary['parameter_reduction_vs_standard'].replace('%', '')),
                "estimated_memory_gb": float(efficiency_summary['estimated_memory_gb']),
                "segformer_style": efficiency_summary['segformer_style']
            })
            
            # Save model as wandb artifact with efficiency info
            artifact_name = f"ultra-efficient-swinunetr-{efficiency_level}"
            if efficiency_summary['segformer_style']:
                artifact_name += "-segformer-style"
            
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(model_name)
            artifact.metadata = efficiency_summary
            wandb.log_artifact(artifact)
        
        # Performance comparison
        print(f"\n🎯 Efficiency Achievement:")
        print(f"   • Parameter reduction: {efficiency_summary['parameter_reduction_vs_standard']}")
        print(f"   • Memory efficiency: ~{efficiency_summary['estimated_memory_gb']} GB")
        print(f"   • Training speed: Enhanced due to smaller model")
        print(f"   • Inference speed: Faster due to efficient architecture")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        
        if "out of memory" in str(e).lower():
            print("1. Switch to 'segformer_style' configuration (ultra-efficient)")
            print("2. Reduce batch_size in your run.py")
            print("3. Reduce sw_batch_size from current to 1")
            print("4. The model should already be very memory efficient!")
            print("5. Check if other processes are using GPU memory")
        elif "checkpoint" in str(e).lower():
            print("1. Check disk space for saving checkpoints")
            print("2. Verify write permissions in ./checkpoints/")
        elif "import" in str(e).lower():
            print("1. Ensure architecture.py contains UltraEfficientSwinUNETR")
            print("2. Check MONAI installation: pip show monai")
            print("3. Verify all imports in pipeline.py")
        else:
            print("1. Check MONAI installation: pip show monai")
            print("2. Verify PyTorch Lightning compatibility")
            print("3. Check dataset format and paths")
            print("4. Ensure architecture.py is properly updated")
        
        raise e
    
    finally:
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
    
    return model


def load_trained_model(checkpoint_path, model_config):
    """Load a trained Ultra-Efficient SwinUNETR model"""
    print(f"📥 Loading trained Ultra-Efficient SwinUNETR from: {checkpoint_path}")
    
    # Create model with same configuration
    model = BrainTumorSegmentation(
        train_loader=None,  # Not needed for inference
        val_loader=None,
        **model_config
    )
    
    # Load weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    model.eval()
    
    # Display efficiency info
    efficiency_summary = model.get_efficiency_summary()
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model info: {efficiency_summary['model_type']}")
    print(f"🎯 Efficiency: {efficiency_summary['efficiency_level']} level")
    print(f"📈 Parameters: {efficiency_summary['parameters_mb']:.2f}M")
    
    return model


def get_model_summary(model):
    """Get detailed model summary with efficiency metrics"""
    efficiency_summary = model.get_efficiency_summary()
    
    # Enhanced summary with efficiency metrics
    summary = {
        "architecture": efficiency_summary["model_type"],
        "efficiency_level": efficiency_summary["efficiency_level"],
        "total_parameters": efficiency_summary["total_parameters"],
        "parameters_mb": efficiency_summary["parameters_mb"],
        "feature_size": efficiency_summary["feature_size"],
        "depths": efficiency_summary["depths"],
        "num_heads": efficiency_summary["num_heads"],
        "decoder_channels": efficiency_summary["decoder_channels"],
        "use_v2": model.hparams.use_v2,
        "segformer_style": efficiency_summary["segformer_style"],
        "parameter_reduction": efficiency_summary["parameter_reduction_vs_standard"],
        "estimated_memory_gb": efficiency_summary["estimated_memory_gb"],
        "efficiency_comparison": {
            "vs_standard_swinunetr": efficiency_summary["parameter_reduction_vs_standard"],
            "vs_segformer3d": "Similar efficiency range (5-25M parameters)",
            "vs_unet3d": "~50-70% fewer parameters",
            "vs_nnunet": "~40-80% fewer parameters"
        }
    }
    
    return summary


def compare_efficiency_levels():
    """Compare different efficiency levels available"""
    print("\n📊 ULTRA-EFFICIENT SWINUNETR EFFICIENCY LEVELS:")
    print("=" * 60)
    
    levels = {
        "segformer_style": {
            "params": "5-8M",
            "memory": "~2GB", 
            "description": "Maximum efficiency, SegFormer3D-like",
            "best_for": "Limited GPU memory, fast inference"
        },
        "ultra": {
            "params": "8-12M",
            "memory": "~2.5GB",
            "description": "Ultra lightweight, minimal parameters",
            "best_for": "Resource-constrained environments"
        },
        "high": {
            "params": "12-18M",
            "memory": "~3GB",
            "description": "High efficiency with good performance",
            "best_for": "Balanced training speed and accuracy"
        },
        "balanced": {
            "params": "18-25M",
            "memory": "~3.5GB",
            "description": "Balanced efficiency and performance",
            "best_for": "Most use cases, good compromise"
        },
        "performance": {
            "params": "25-35M",
            "memory": "~4GB",
            "description": "Performance-focused efficiency",
            "best_for": "Best accuracy while staying efficient"
        }
    }
    
    for level, info in levels.items():
        print(f"\n🎯 {level.upper()}:")
        print(f"   Parameters: {info['params']}")
        print(f"   Memory: {info['memory']}")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {info['best_for']}")
    
    print(f"\n📋 Reference comparison:")
    print(f"   • Standard MONAI SwinUNETR: ~62M parameters")
    print(f"   • SegFormer3D: ~8-15M parameters")
    print(f"   • UNet3D: ~30-40M parameters")
    print(f"   • nnU-Net: ~31M parameters")


def benchmark_efficiency(model, input_shape=(1, 4, 128, 128, 128)):
    """Benchmark model efficiency metrics"""
    print(f"\n⚡ Benchmarking Ultra-Efficient SwinUNETR...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create sample input
    x = torch.randn(input_shape).to(device)
    
    # Measure memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure inference time
    import time
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Actual timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / 10
    
    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        peak_memory = "N/A (CPU)"
    
    # Get efficiency summary
    efficiency_summary = model.get_efficiency_summary()
    
    benchmark_results = {
        "model_type": efficiency_summary["model_type"],
        "efficiency_level": efficiency_summary["efficiency_level"],
        "parameters": f"{efficiency_summary['parameters_mb']:.2f}M",
        "avg_inference_time": f"{avg_inference_time:.4f}s",
        "peak_memory_usage": f"{peak_memory:.2f}GB" if isinstance(peak_memory, float) else peak_memory,
        "throughput": f"{1/avg_inference_time:.2f} samples/sec",
        "parameter_reduction": efficiency_summary["parameter_reduction_vs_standard"],
        "output_shape": list(output.shape)
    }
    
    print(f"📊 Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    return benchmark_results


# Utility function for easy model creation in research/inference
def create_efficient_model_for_inference(efficiency_level="balanced", checkpoint_path=None):
    """Create an efficient model optimized for inference"""
    
    # Default configurations for inference
    inference_configs = {
        "segformer_style": {
            "efficiency_level": "ultra",
            "use_segformer_style": True,
            "feature_size": 16,
        },
        "ultra": {
            "efficiency_level": "ultra", 
            "feature_size": 16,
            "depths": (1, 1, 1, 1),
            "num_heads": (1, 2, 4, 8),
        },
        "balanced": {
            "efficiency_level": "balanced",
            "feature_size": 24,
            "depths": (1, 1, 2, 1),
            "num_heads": (2, 4, 8, 16),
        }
    }
    
    config = inference_configs.get(efficiency_level, inference_configs["balanced"])
    
    model = BrainTumorSegmentation(
        train_loader=None,
        val_loader=None,
        **config
    )
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    return model