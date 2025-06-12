# trainer.py
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from .pipeline import BrainTumorSegmentation


# Using standard EarlyStopping - no custom implementation needed


def setup_training(train_ds, val_ds, args):
    """Setup training with MONAI SwinUNETR configurations"""
    
    # Enhanced data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers,
        drop_last=True  # Ensure consistent batch sizes
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers
    )

    # Standard early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.001,
        patience=args.early_stopping_patience,
        verbose=True,
        mode='max'
    )
    
    # Model checkpointing with better naming
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="monai-swinunetr-{epoch:02d}-{val_mean_dice:.4f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False  # Save after validation
    )
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Timer callback
    timer_callback = Timer(duration="00:12:00:00")  # 12 hours

    # Initialize wandb with MONAI-specific tracking
    wandb.init(
        project="brain-tumor-segmentation-monai", 
        name=f"monai-swinunetr-v2-{args.feature_size}feat",
        config={
            "architecture": "MONAI_SwinUNETR_V2",
            "feature_size": args.feature_size,
            "depths": args.depths,
            "num_heads": args.num_heads,
            "use_checkpoint": args.use_checkpoint,
            "use_v2": args.use_v2,
            "norm_name": args.norm_name,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
            "epochs": args.epochs,
            "drop_rate": args.drop_rate,
            "attn_drop_rate": args.attn_drop_rate,
            "dropout_path_rate": args.dropout_path_rate
        }
    )
    wandb_logger = WandbLogger()

    # Initialize MONAI SwinUNETR model
    model = BrainTumorSegmentation(
        train_loader,
        val_loader,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        feature_size=args.feature_size,
        depths=args.depths,
        num_heads=args.num_heads,
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
        # Legacy parameters (will be ignored)
        img_size=getattr(args, 'img_size', 128),
        embed_dim=getattr(args, 'embed_dim', None),
        window_size=getattr(args, 'window_size', None),
        mlp_ratio=getattr(args, 'mlp_ratio', None),
        decoder_embed_dim=getattr(args, 'decoder_embed_dim', None),
        patch_size=getattr(args, 'patch_size', None)
    )

    # Setup trainer with MONAI optimizations
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if args.use_amp else '32',
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=5,  # More frequent logging for MONAI
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
        benchmark=True,  # Optimize for MONAI's consistent operations
        enable_checkpointing=True,
        enable_model_summary=True,
        sync_batchnorm=False,  # Not needed for single GPU
        enable_progress_bar=True
    )

    return model, trainer, train_loader, val_loader


def train_model(model, trainer, train_loader, val_loader):
    """Train the MONAI SwinUNETR model with enhanced monitoring"""
    print("🚀 Starting MONAI SwinUNETR training...")
    
    # Log model architecture details
    total_params = model.model.count_parameters()
    print(f"📊 Training MONAI SwinUNETR with {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Log model configuration
    print(f"🔧 Model Configuration:")
    print(f"   Feature size: {model.hparams.feature_size}")
    print(f"   Depths: {model.hparams.depths}")
    print(f"   Num heads: {model.hparams.num_heads}")
    print(f"   Use V2: {model.hparams.use_v2}")
    print(f"   Use checkpoint: {model.hparams.use_checkpoint}")
    print(f"   Norm: {model.hparams.norm_name}")
    print(f"   Dropout rates: {model.hparams.drop_rate}/{model.hparams.attn_drop_rate}/{model.hparams.dropout_path_rate}")
    
    # Start training
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Final results
        print(f"\n✅ MONAI SwinUNETR training completed successfully!")
        print(f"📊 Results:")
        print(f"   Best validation Dice: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}")
        
        if len(model.metric_values_tc) > 0:
            print(f"   Final individual metrics:")
            print(f"     TC (Tumor Core): {model.metric_values_tc[-1]:.4f}")
            print(f"     WT (Whole Tumor): {model.metric_values_wt[-1]:.4f}")
            print(f"     ET (Enhancing Tumor): {model.metric_values_et[-1]:.4f}")
        
        # Save final model with better naming
        final_model_path = f"final_monai_swinunetr_feat{model.hparams.feature_size}.pth"
        torch.save(model.model.state_dict(), final_model_path)
        print(f"💾 Final model saved as: {final_model_path}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "final_best_dice": model.best_metric,
                "final_tc": model.metric_values_tc[-1] if len(model.metric_values_tc) > 0 else 0,
                "final_wt": model.metric_values_wt[-1] if len(model.metric_values_wt) > 0 else 0,
                "final_et": model.metric_values_et[-1] if len(model.metric_values_et) > 0 else 0,
                "total_parameters": total_params
            })
            
            # Save model as wandb artifact
            artifact = wandb.Artifact("monai-swinunetr-model", type="model")
            artifact.add_file(final_model_path)
            wandb.log_artifact(artifact)
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        
        if "out of memory" in str(e).lower():
            print("1. Reduce batch_size in your run.py")
            print("2. Switch to 'ultra_lightweight' configuration")
            print("3. Reduce sw_batch_size from 2 to 1")
            print("4. Enable use_checkpoint=True (should already be enabled)")
        elif "checkpoint" in str(e).lower():
            print("1. Check disk space for saving checkpoints")
            print("2. Verify write permissions in ./checkpoints/")
        else:
            print("1. Check MONAI installation: pip show monai")
            print("2. Verify PyTorch Lightning compatibility")
            print("3. Check dataset format and paths")
        
        raise e
    
    finally:
        # Cleanup
        if wandb.run is not None:
            wandb.finish()
    
    return model


def load_trained_model(checkpoint_path, model_config):
    """Load a trained MONAI SwinUNETR model"""
    print(f"📥 Loading trained MONAI SwinUNETR from: {checkpoint_path}")
    
    # Create model with same configuration
    model = BrainTumorSegmentation(
        train_loader=None,  # Not needed for inference
        val_loader=None,
        **model_config
    )
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model


def get_model_summary(model):
    """Get detailed model summary"""
    total_params = model.model.count_parameters()
    
    summary = {
        "total_parameters": total_params,
        "parameters_mb": total_params / 1e6,
        "feature_size": model.hparams.feature_size,
        "depths": model.hparams.depths,
        "num_heads": model.hparams.num_heads,
        "use_v2": model.hparams.use_v2,
        "architecture": "MONAI SwinUNETR V2"
    }
    
    return summary