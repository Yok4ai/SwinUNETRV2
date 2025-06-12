#trainer.py
import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loggers import WandbLogger
import wandb
from .architecture import BrainTumorSegmentation

def setup_training(train_ds, val_ds, args):
    # Data loaders
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

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.00,
        patience=args.early_stopping_patience,
        verbose=True,
        mode='max'
    )

    # Timer callback
    timer_callback = Timer(duration="00:11:00:00")

    # Initialize wandb
    wandb.init(project="brain-tumor-segmentation", name="swinunetr-v2-brats23")
    wandb_logger = WandbLogger()

    # Initialize model
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

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if args.use_amp else '32',
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, timer_callback],
        limit_val_batches=5,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
    )

    return model, trainer, train_loader, val_loader

def train_model(model, trainer, train_loader, val_loader):
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Train completed, best_metric: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}.") 