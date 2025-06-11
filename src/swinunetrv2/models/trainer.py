import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loggers import WandbLogger
import wandb
from .architecture import BrainTumorSegmentation

def setup_training(train_ds, val_ds, max_epochs=50):
    # Data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.00,
        patience=15,
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
        max_epochs=max_epochs,
        learning_rate=2e-3,
        img_size=128,
        feature_size=48,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        decoder_embed_dim=256,
        patch_size=4,
        weight_decay=1e-4,
        warmup_epochs=5,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        roi_size=(128, 128, 128),
        sw_batch_size=2,
        overlap=0.25
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed' if True else '32',
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, timer_callback],
        limit_val_batches=5,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )

    return model, trainer, train_loader, val_loader

def train_model(model, trainer, train_loader, val_loader):
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Train completed, best_metric: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}.") 