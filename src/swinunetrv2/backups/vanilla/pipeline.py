import os
import time
import torch
import pytorch_lightning as pl
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import PersistentDataset, list_data_collate, decollate_batch, DataLoader, load_decathlon_datalist, CacheDataset
from monai.inferers import sliding_window_inference
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from monai.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from torch.cuda.amp import GradScaler
import wandb
from pytorch_lightning.loggers import WandbLogger

from .architecture import SwinUNETR


class BrainTumorSegmentation(pl.LightningModule):
    def __init__(self, 
        train_loader,
        val_loader,
        max_epochs=100,
        val_interval=1, 
        learning_rate=1e-4,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        patch_size=(2, 2, 2),
        weight_decay=1e-5,
        warmup_epochs=5,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        roi_size=(128, 128, 128),
        sw_batch_size=4,
        overlap=0.5
        ):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,  # Enable SwinUNETR-V2!
            spatial_dims=3,
            depths=depths,
            num_heads=num_heads,
            norm_name="instance",
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=0.0,
            downsample="mergingv2"  # Use improved merging for V2
        )
        # Loss and metrics
        self.loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Best metric tracking
        self.best_metric = -1

        # Training metrics
        self.avg_train_loss_values = []
        self.train_loss_values = []
        self.train_metric_values = []
        self.train_metric_values_tc = []
        self.train_metric_values_wt = []
        self.train_metric_values_et = []

        # Validation metrics
        self.avg_val_loss_values = []
        self.epoch_loss_values = []
        self.metric_values = []
        self.metric_values_tc = []
        self.metric_values_wt = []
        self.metric_values_et = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]

        # Calculate Train Loss
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        
        # Log the Train Loss
        self.log("train_loss", loss, prog_bar=True)

        # Apply sigmoid and threshold (same as validation)
        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
        
        # Compute Dice
        self.dice_metric(y_pred=outputs, y=labels)
        self.dice_metric_batch(y_pred=outputs, y=labels)

        # Log Train Dice 
        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True)

        # Store Mean Dice
        self.train_metric_values.append(train_dice)

        # Store the individual dice
        metric_batch = self.dice_metric_batch.aggregate()
        self.train_metric_values_tc.append(metric_batch[0].item())
        self.train_metric_values_wt.append(metric_batch[1].item())
        self.train_metric_values_et.append(metric_batch[2].item())

        # Log the individual dice metrics
        self.log("train_tc", metric_batch[0].item(), prog_bar=True)
        self.log("train_wt", metric_batch[1].item(), prog_bar=True)
        self.log("train_et", metric_batch[2].item(), prog_bar=True)

        # Reset metrics for the next epoch
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.logged_metrics["train_loss"].item()
        self.train_loss_values.append(train_loss)
        
        # Calculate and store average loss per epoch
        avg_train_loss = sum(self.train_loss_values) / len(self.train_loss_values)
        self.log("avg_train_loss", avg_train_loss, prog_bar=True)
        self.avg_train_loss_values.append(avg_train_loss)

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        val_outputs = sliding_window_inference(
            val_inputs, roi_size=(96, 96, 96), sw_batch_size=1, predictor=self.model, overlap=0.5
        )
        
        # Compute loss
        val_loss = self.loss_function(val_outputs, val_labels)
        # Log validation loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]    
        # Compute Dice
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        self.dice_metric_batch(y_pred=val_outputs, y=val_labels)
    
        # Log validation Dice
        val_dice = self.dice_metric.aggregate().item()
        self.log("val_mean_dice", val_dice, prog_bar=True)
    
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        # Store Dice Mean
        val_dice = self.dice_metric.aggregate().item()
        self.metric_values.append(val_dice)

        # Store Validation Loss 
        val_loss = self.trainer.logged_metrics["val_loss"].item()
        self.epoch_loss_values.append(val_loss)

        # Calculate and Store avg val loss values
        avg_val_loss = sum(self.epoch_loss_values) / len(self.epoch_loss_values)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.avg_val_loss_values.append(avg_val_loss)

        # Store Individual Dice
        metric_batch = self.dice_metric_batch.aggregate()
        self.metric_values_tc.append(metric_batch[0].item())
        self.metric_values_wt.append(metric_batch[1].item())
        self.metric_values_et.append(metric_batch[2].item())

        # Log validation metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mean_dice", val_dice, prog_bar=True)
        self.log("val_tc", metric_batch[0].item(), prog_bar=True)
        self.log("val_wt", metric_batch[1].item(), prog_bar=True)
        self.log("val_et", metric_batch[2].item(), prog_bar=True)
    
        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_metric_model_swinunetr_v2.pth")
            self.log("best_metric", self.best_metric)
    
        # Reset metrics for the next epoch
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def on_train_end(self):
        # Print the best metric and epoch along with individual Dice scores at the end of training
        print(f"Train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}, "
              f"tc: {self.metric_values_tc[-1]:.4f}, "
              f"wt: {self.metric_values_wt[-1]:.4f}, "
              f"et: {self.metric_values_et[-1]:.4f}.")

    def configure_optimizers(self):
        # SwinUNETR-V2 benefits from slightly higher learning rate
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]