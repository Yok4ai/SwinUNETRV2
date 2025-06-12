#trainer.py
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from .pipeline import BrainTumorSegmentation

def setup_training(train_ds, val_ds, args):
    """
    Setup training with direct parameter passing from args
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        args: Argument namespace containing all configuration parameters
    
    Returns:
        model: BrainTumorSegmentation model
        trainer: PyTorch Lightning trainer
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    
    # Data loaders using parameters from args
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        persistent_workers=args.persistent_workers
    )

    # Early stopping callback using parameters from args
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.001,
        patience=args.early_stopping_patience,
        verbose=True,
        mode='max'
    )

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='swinunetr-v2-{epoch:02d}-{val_mean_dice:.4f}',
        save_top_k=3,
        monitor='val_mean_dice',
        mode='max',
        save_last=True
    )

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="brain-tumor-segmentation",
        name="swinunetr-v2-brats23",
        log_model=True
    )

    # Initialize model with args - much cleaner parameter passing
    model = BrainTumorSegmentation(
        args=args,
        train_loader=train_loader,  # Optional, for backward compatibility
        val_loader=val_loader       # Optional, for backward compatibility
    )

    # Setup trainer with parameters from args
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        strategy="auto",  # Let PyTorch Lightning choose
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if args.use_amp else '32',
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
        # Memory optimization settings
        deterministic=False,  # Allow for faster training
        benchmark=True,  # Optimize cudnn for consistent input sizes
        enable_model_summary=True,
        enable_checkpointing=True,
    )

    return model, trainer, train_loader, val_loader

def train_model(model, trainer, train_loader, val_loader):
    """
    Train the model using the provided trainer and data loaders
    
    Args:
        model: BrainTumorSegmentation model
        trainer: PyTorch Lightning trainer
        train_loader: Training data loader
        val_loader: Validation data loader
    
    Returns:
        best_metric: Best validation metric achieved during training
    """
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"Train completed, best_metric: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}.")
        return model.best_metric
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise e