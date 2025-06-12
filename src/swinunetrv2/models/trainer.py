# hybrid_trainer.py - Trainer for Hybrid SwinUNETR-SegFormer3D
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from .pipeline import HybridBrainTumorSegmentation


def setup_hybrid_training(train_ds, val_ds, args):
    """Setup training for Hybrid SwinUNETR-SegFormer3D"""
    
    # Data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory
    )

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.001,
        patience=args.early_stopping_patience,
        verbose=True,
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="hybrid-swinunetr-segformer-{epoch:02d}-{val_mean_dice:.4f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Wandb logging
    wandb.init(
        project="brain-tumor-hybrid-swinunetr-segformer3d",
        name=f"hybrid-{args.efficiency_level}-{args.decoder_embedding_dim}dim"
    )
    wandb_logger = WandbLogger()

    # Initialize hybrid model
    model = HybridBrainTumorSegmentation(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        # Hybrid-specific parameters
        efficiency_level=args.efficiency_level,
        decoder_embedding_dim=args.decoder_embedding_dim,
        use_segformer_decoder=args.use_segformer_decoder,
        # SwinUNETR backbone parameters
        feature_size=args.feature_size,
        depths=args.depths,
        num_heads=args.num_heads,
        # Training parameters
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.dropout_path_rate,
        decoder_dropout=args.decoder_dropout,
        # Inference parameters
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        use_mixed_precision=args.use_amp,
        norm_name=args.norm_name,
        use_checkpoint=args.use_checkpoint,
        use_v2=args.use_v2
    )

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if args.use_amp else '32',
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=5,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        benchmark=True,
        enable_model_summary=True
    )

    return model, trainer, train_loader, val_loader


def train_hybrid_model(model, trainer, train_loader, val_loader):
    """Train the Hybrid SwinUNETR-SegFormer3D model"""
    
    # Get model info
    model_info = model.get_model_info()
    total_params = model_info["total_parameters"]
    
    print(f"🚀 Starting {model_info['architecture']} training...")
    print(f"📊 Model Configuration:")
    print(f"   Architecture: {model_info['architecture']}")
    print(f"   Efficiency level: {model_info['efficiency_level']}")
    print(f"   Backbone: {model_info['backbone']}")
    print(f"   Decoder: {model_info['decoder']}")
    print(f"   Parameters: {total_params:,} ({model_info['parameters_mb']:.2f}M)")
    print(f"   Decoder embedding dim: {model_info['decoder_embedding_dim']}")
    print(f"   V2 merging: {model_info['use_v2_merging']}")
    print(f"   Parameter reduction: {model_info['parameter_reduction_vs_standard']}")
    
    try:
        # Start training
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Training completed successfully
        print(f"✅ Hybrid training completed!")
        print(f"📊 Results:")
        print(f"   Best Dice: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}")
        print(f"   Architecture: {model_info['architecture']}")
        print(f"   Total parameters: {total_params:,} ({model_info['parameters_mb']:.2f}M)")
        
        if len(model.metric_values_tc) > 0:
            print(f"   Final individual metrics:")
            print(f"     TC (Tumor Core): {model.metric_values_tc[-1]:.4f}")
            print(f"     WT (Whole Tumor): {model.metric_values_wt[-1]:.4f}")
            print(f"     ET (Enhancing Tumor): {model.metric_values_et[-1]:.4f}")
        
        # Save final model
        model_name = f"final_hybrid_swinunetr_segformer_{model_info['efficiency_level']}.pth"
        torch.save(model.model.state_dict(), model_name)
        print(f"💾 Final hybrid model saved as: {model_name}")
        
        # Enhanced wandb logging
        if wandb.run is not None:
            wandb.log({
                "final_best_dice": model.best_metric,
                "final_tc": model.metric_values_tc[-1] if len(model.metric_values_tc) > 0 else 0,
                "final_wt": model.metric_values_wt[-1] if len(model.metric_values_wt) > 0 else 0,
                "final_et": model.metric_values_et[-1] if len(model.metric_values_et) > 0 else 0,
                "total_parameters": total_params,
                "efficiency_level": model_info['efficiency_level'],
                "decoder_type": model_info['decoder'],
                "parameter_reduction_percent": float(model_info['parameter_reduction_vs_standard'].replace('%', '')),
                "decoder_embedding_dim": model_info['decoder_embedding_dim']
            })
        
        # Architecture comparison
        print(f"\n🎯 Hybrid Architecture Benefits:")
        print(f"   ✅ SwinUNETR V2 backbone: Proven encoder performance")
        print(f"   ✅ SegFormer3D decoder: Lightweight and efficient")
        print(f"   ✅ V2 merging: Better feature integration")
        print(f"   ✅ Parameter efficiency: {model_info['parameter_reduction_vs_standard']} reduction")
        print(f"   ✅ Memory efficient: SegFormer-style MLP decoder")
        
    except Exception as e:
        print(f"❌ Hybrid training failed: {e}")
        
        # Specific troubleshooting for hybrid model
        if "out of memory" in str(e).lower():
            print("\n🔧 Hybrid model memory optimization:")
            print("1. The hybrid should already be more efficient!")
            print("2. Try efficiency_level='light' (smallest config)")
            print("3. Reduce decoder_embedding_dim from 128 to 96")
            print("4. Reduce batch_size or sw_batch_size")
        elif "shape" in str(e).lower() or "dimension" in str(e).lower():
            print("\n🔧 Shape mismatch troubleshooting:")
            print("1. Check that SwinUNETR encoder outputs match expected dimensions")
            print("2. Verify SegFormer decoder input dimensions")
            print("3. The hybrid forward pass might need adjustment")
        else:
            print("\n🔧 General troubleshooting:")
            print("1. Verify MONAI SwinUNETR is properly imported")
            print("2. Check that hybrid_architecture.py is in the correct path")
            print("3. Ensure all dependencies are installed")
        
        raise e
    
    finally:
        if wandb.run is not None:
            wandb.finish()
    
    return model


def load_hybrid_model(checkpoint_path, model_config):
    """Load a trained Hybrid SwinUNETR-SegFormer3D model"""
    print(f"📥 Loading Hybrid SwinUNETR-SegFormer3D from: {checkpoint_path}")
    
    # Create model with same configuration
    model = HybridBrainTumorSegmentation(
        train_loader=None,
        val_loader=None,
        **model_config
    )
    
    # Load weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    model.eval()
    
    # Display model info
    model_info = model.get_model_info()
    print(f"✅ Hybrid model loaded successfully!")
    print(f"📊 Architecture: {model_info['architecture']}")
    print(f"🎯 Efficiency: {model_info['efficiency_level']} level")
    print(f"📈 Parameters: {model_info['parameters_mb']:.2f}M")
    print(f"🔗 Decoder: {model_info['decoder']}")
    
    return model


# Easy configuration helpers
def get_light_hybrid_config():
    """Lightweight hybrid config for limited resources"""
    return {
        "efficiency_level": "light",
        "decoder_embedding_dim": 96,
        "batch_size": 6,
        "accumulate_grad_batches": 2,
        "learning_rate": 8e-4,
        "sw_batch_size": 3,
    }

def get_balanced_hybrid_config():
    """Balanced hybrid config - recommended default"""
    return {
        "efficiency_level": "balanced", 
        "decoder_embedding_dim": 128,
        "batch_size": 4,
        "accumulate_grad_batches": 3,
        "learning_rate": 5e-4,
        "sw_batch_size": 2,
    }

def get_performance_hybrid_config():
    """Performance-focused hybrid config"""
    return {
        "efficiency_level": "performance",
        "decoder_embedding_dim": 192,
        "batch_size": 3,
        "accumulate_grad_batches": 4,
        "learning_rate": 3e-4,
        "sw_batch_size": 2,
    }

# Add alias for backward compatibility
setup_training = setup_hybrid_training