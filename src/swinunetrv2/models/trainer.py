# improved_trainer.py
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from .architecture import BrainTumorSegmentation


class CustomEarlyStopping(EarlyStopping):
    """Custom early stopping with better patience strategy"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_score_history = []
    
    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.logged_metrics.get(self.monitor)
        if current_score is not None:
            self.best_score_history.append(current_score.item())
            
            # Only stop if we haven't improved in patience epochs AND
            # we've been training for at least 15 epochs
            if len(self.best_score_history) > 15:
                super().on_validation_end(trainer, pl_module)


def setup_training(train_ds, val_ds, args):
    """Setup training with configurations"""
    
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

    # Improved callbacks
    early_stop_callback = CustomEarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.005,  # Smaller threshold for improvement
        patience=20,  # More patience
        verbose=True,
        mode='max'
    )
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="swinunetr-{epoch:02d}-{val_mean_dice:.3f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1
    )
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Timer callback
    timer_callback = Timer(duration="00:12:00:00")  # 12 hours

    # Initialize wandb with better tracking
    wandb.init(
        project="brain-tumor-segmentation", 
        name="swinunetr-v3-brats23",
        config={
            "architecture": "LightweightSwinUNETR",
            "embed_dim": args.embed_dim,
            "depths": args.depths,
            "num_heads": args.num_heads,
            "decoder_embed_dim": args.decoder_embed_dim,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs
        }
    )
    wandb_logger = WandbLogger()

    # Initialize model with hyperparameters
    model = BrainTumorSegmentation(
        train_loader,
        val_loader,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        feature_size=args.feature_size,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        decoder_embed_dim=args.decoder_embed_dim,
        patch_size=args.patch_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap
    )

    # Setup trainer with configurations
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if args.use_amp else '32',
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
        callbacks=[
            early_stop_callback, 
            checkpoint_callback, 
            lr_monitor, 
            timer_callback
        ],
        limit_val_batches=args.limit_val_batches if hasattr(args, 'limit_val_batches') else None,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches if hasattr(args, 'accumulate_grad_batches') else 1,
        deterministic=False,  # Allow for faster training
        benchmark=True,  # Optimize cudnn for consistent input sizes
    )

    return model, trainer, train_loader, val_loader


def train_model(model, trainer, train_loader, val_loader):
    """Train the model with monitoring"""
    print("🚀 Starting training...")
    
    # Log model architecture details
    total_params = model.model.count_parameters()
    print(f"📊 Training model with {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Start training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Final results
    print(f"✅ Training completed!")
    print(f"   Best metric: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}")
    print(f"   Final TC: {model.metric_values_tc[-1]:.4f}")
    print(f"   Final WT: {model.metric_values_wt[-1]:.4f}")
    print(f"   Final ET: {model.metric_values_et[-1]:.4f}")
    
    # Save final model
    torch.save(model.model.state_dict(), "final_swinunetr.pth")
    
    return model
