import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loggers import WandbLogger
import wandb
from .architecture import BrainTumorSegmentation

def setup_training(train_ds, val_ds, max_epochs=30):
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=False)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_mean_dice",
        min_delta=0.00,
        patience=7,
        verbose=True,
        mode='max'
    )

    # Timer callback
    timer_callback = Timer(duration="00:11:00:00")

    # Initialize wandb
    wandb.init(project="brain-tumor-segmentation", name="swinunetr-v2-brats23")
    wandb_logger = WandbLogger()

    # Initialize model
    model = BrainTumorSegmentation(train_loader, val_loader, max_epochs=max_epochs)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="gpu",
        precision='16-mixed',
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