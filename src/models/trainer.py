#trainer.py
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.models.pipeline import BrainTumorSegmentation

def setup_training(train_loader, val_loader, args):
    """
    Setup training with direct parameter passing from args
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        args: Argument namespace containing all configuration parameters (use_class_weights should be set in the namespace)
    Returns:
        model: BrainTumorSegmentation model
        trainer: PyTorch Lightning trainer
    """
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
        filename='swinunetr-{epoch:02d}-{val_mean_dice:.4f}',
        monitor='val_mean_dice',
        mode='max',
        save_top_k=3,
        save_last=True
    )

    # Learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project="brain-tumor-segmentation",
        name="swinunetr-v2-experimental",
        log_model=False
    )

    # Initialize model
    model = BrainTumorSegmentation(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        learning_rate=args.learning_rate,
        feature_size=args.feature_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        use_v2=args.use_v2,
        depths=args.depths,
        num_heads=args.num_heads,
        downsample=args.downsample,
        use_class_weights=getattr(args, 'use_class_weights', True),
        loss_type=getattr(args, 'loss_type', 'dice'),
        use_modality_attention=getattr(args, 'use_modality_attention', False),
        overlap=getattr(args, 'overlap', 0.7),
        class_weights=getattr(args, 'class_weights', (3.0, 1.0, 5.0)),
        threshold=getattr(args, 'threshold', 0.5),
        optimizer_betas=getattr(args, 'optimizer_betas', (0.9, 0.999)),
        optimizer_eps=getattr(args, 'optimizer_eps', 1e-8),
        # New loss parameters
        tversky_alpha=getattr(args, 'tversky_alpha', 0.5),
        tversky_beta=getattr(args, 'tversky_beta', 0.5),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
        focal_alpha=getattr(args, 'focal_alpha', None),
        gdl_weight_type=getattr(args, 'gdl_weight_type', 'square'),
        gdl_lambda=getattr(args, 'gdl_lambda', 1.0),
        hausdorff_alpha=getattr(args, 'hausdorff_alpha', 2.0),
        lambda_dice=getattr(args, 'lambda_dice', 1.0),
        lambda_focal=getattr(args, 'lambda_focal', 1.0),
        lambda_tversky=getattr(args, 'lambda_tversky', 1.0),
        lambda_hausdorff=getattr(args, 'lambda_hausdorff', 1.0),
        # Adaptive loss scheduling parameters
        use_adaptive_scheduling=getattr(args, 'use_adaptive_scheduling', False),
        adaptive_schedule_type=getattr(args, 'adaptive_schedule_type', 'linear'),
        structure_epochs=getattr(args, 'structure_epochs', 30),
        boundary_epochs=getattr(args, 'boundary_epochs', 50),
        schedule_start_epoch=getattr(args, 'schedule_start_epoch', 10),
        min_loss_weight=getattr(args, 'min_loss_weight', 0.1),
        max_loss_weight=getattr(args, 'max_loss_weight', 2.0),
        # Warm restart parameters
        use_warm_restarts=getattr(args, 'use_warm_restarts', False),
        restart_period=getattr(args, 'restart_period', 20),
        restart_mult=getattr(args, 'restart_mult', 1),
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        strategy=args.strategy,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        enable_checkpointing=args.enable_checkpointing,
        benchmark=args.benchmark,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        profiler=args.profiler
    )

    return model, trainer

def train_model(model, trainer, train_loader, val_loader, resume_from_checkpoint=None):
    """
    Train the model using the provided trainer and data loaders
    
    Args:
        model: BrainTumorSegmentation model
        trainer: PyTorch Lightning trainer
        train_loader: Training data loader
        val_loader: Validation data loader
        resume_from_checkpoint: Path to checkpoint file to resume from
    """
    try:
        # Start training
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_from_checkpoint
        )
        
        # Log final metrics
        print("\n=== Training Complete ===")
        print(f"Best validation dice score: {model.best_metric:.4f}")
        print(f"Best epoch: {model.best_metric_epoch}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e