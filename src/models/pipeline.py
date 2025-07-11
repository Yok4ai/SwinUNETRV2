import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
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
import math
# from .swinunetr import SwinUNETR
from monai.networks.nets import SwinUNETR


# Remove ModalityAttentionModule and EnhancedSwinUNETR classes

class BrainTumorSegmentation(pl.LightningModule):
    def __init__(self, train_loader, val_loader, max_epochs=100,
                 val_interval=1, learning_rate=1e-4, feature_size=48,
                 weight_decay=1e-5, warmup_epochs=10, roi_size=(96, 96, 96),
                 sw_batch_size=2, use_v2=True, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), downsample="mergingv2",
                 use_class_weights=True,
                 loss_type='hybrid',
                 ):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Only use vanilla SwinUNETR
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=self.hparams.feature_size,
            use_checkpoint=True,
            use_v2=self.hparams.use_v2,
            spatial_dims=3,
            depths=self.hparams.depths,
            num_heads=self.hparams.num_heads,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            downsample=self.hparams.downsample,
            
        )
        
        # Class weights based on BraTS imbalance: ET (most rare) > TC > WT
        if self.hparams.use_class_weights:
            # Higher weights for more imbalanced classes
            class_weights = torch.tensor([1.0, 3.0, 5.0])  # Background, WT, TC, ET
        else:
            class_weights = None
            
        # Loss functions with class weighting
        self.loss_type = loss_type

        self.dice_loss = DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
            to_onehot_y=False, sigmoid=True
        )
        self.ce_loss = DiceCELoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
            to_onehot_y=False, sigmoid=True
        )
        self.focal_loss = FocalLoss(
            gamma=2.0, weight=class_weights, reduction='mean'
        )
        
        # Standard Dice Loss Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.jaccard_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        
        self.best_metric = -1
        self.train_loader = train_loader
        self.val_loader = val_loader

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

    def compute_loss(self, outputs, labels):
        """Compute loss based on loss_type: 'hybrid' (DiceCE+Focal) or 'dice' (Dice only)"""
        if self.loss_type == 'hybrid':
            dice_ce_loss = self.ce_loss(outputs, labels)
            focal_loss = self.focal_loss(outputs, labels)
            total_loss = 0.6 * dice_ce_loss + 0.4 * focal_loss
            return total_loss
        else:
            return self.dice_loss(outputs, labels)

    def compute_metrics(self, outputs, labels):
        """Compute mean precision, recall, and F1 score for the batch."""
        # outputs, labels: (B, C, D, H, W)
        outputs = torch.stack(outputs) if isinstance(outputs, list) else outputs
        labels = torch.stack(labels) if isinstance(labels, list) else labels
        outputs = outputs.float()
        labels = labels.float()
        # Flatten all but batch
        outputs = outputs.view(outputs.size(0), -1)
        labels = labels.view(labels.size(0), -1)
        # True Positives, False Positives, False Negatives
        tp = (outputs * labels).sum(dim=1)
        fp = (outputs * (1 - labels)).sum(dim=1)
        fn = ((1 - outputs) * labels).sum(dim=1)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision.mean().item(), recall.mean().item(), f1.mean().item()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]

        # Calculate Train Loss with hybrid approach
        outputs = self(inputs)
        loss = self.compute_loss(outputs, labels)
        
        # Log the Train Loss
        self.log("train_loss", loss, prog_bar=True)

        # Apply sigmoid and threshold
        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
        
        # Compute Dice
        self.dice_metric(y_pred=outputs, y=labels)
        self.dice_metric_batch(y_pred=outputs, y=labels)
        self.jaccard_metric(y_pred=outputs, y=labels)
        # Compute Hausdorff
        self.hausdorff_metric(y_pred=outputs, y=labels)
        hausdorff_values = self.hausdorff_metric.aggregate(reduction='none')
        if not isinstance(hausdorff_values, torch.Tensor):
            hausdorff_values = torch.tensor(hausdorff_values)
        valid = torch.isfinite(hausdorff_values)
        if valid.any():
            train_hausdorff = hausdorff_values[valid].mean().item()
        else:
            train_hausdorff = float('nan')
        self.log("train_hausdorff", train_hausdorff, prog_bar=True)

        # Log Train Dice 
        train_dice = self.dice_metric.aggregate().item()
        train_iou = self.jaccard_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True)
        self.log("train_mean_iou", train_iou, prog_bar=True)

        # Compute and log mean precision, recall, F1
        precision, recall, f1 = self.compute_metrics(outputs, decollate_batch(labels))
        self.log("train_mean_precision", precision, prog_bar=True)
        self.log("train_mean_recall", recall, prog_bar=True)
        self.log("train_mean_f1", f1, prog_bar=True)

        # Store metrics
        self.train_metric_values.append(train_dice)
        metric_batch = self.dice_metric_batch.aggregate()
        self.train_metric_values_tc.append(metric_batch[0].item())
        self.train_metric_values_wt.append(metric_batch[1].item())
        self.train_metric_values_et.append(metric_batch[2].item())

        # Log individual dice metrics
        self.log("train_tc", metric_batch[0].item(), prog_bar=True)
        self.log("train_wt", metric_batch[1].item(), prog_bar=True)
        self.log("train_et", metric_batch[2].item(), prog_bar=True)

        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.jaccard_metric.reset()
        self.hausdorff_metric.reset()

        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.logged_metrics["train_loss"].item()
        self.train_loss_values.append(train_loss)
        
        avg_train_loss = sum(self.train_loss_values) / len(self.train_loss_values)
        self.log("avg_train_loss", avg_train_loss, prog_bar=True, sync_dist=True)
        self.avg_train_loss_values.append(avg_train_loss)

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        
        # Multiple overlapping predictions for better accuracy
        roi_size = (96, 96, 96)
        
        # Original prediction
        val_outputs = sliding_window_inference(
            val_inputs, roi_size=roi_size, sw_batch_size=1, 
            predictor=self.model, overlap=0.5  # Higher overlap
        )
        
        # Compute loss with hybrid approach
        val_loss = self.compute_loss(val_outputs, val_labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]    
        
        # Compute Dice
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        self.dice_metric_batch(y_pred=val_outputs, y=val_labels)
        self.jaccard_metric(y_pred=val_outputs, y=val_labels)
        self.hausdorff_metric(y_pred=val_outputs, y=val_labels)

        val_hausdorff_values = self.hausdorff_metric.aggregate(reduction='none')
        if not isinstance(val_hausdorff_values, torch.Tensor):
            val_hausdorff_values = torch.tensor(val_hausdorff_values)
        val_valid = torch.isfinite(val_hausdorff_values)
        if val_valid.any():
            val_hausdorff = val_hausdorff_values[val_valid].mean().item()
        else:
            val_hausdorff = float('nan')
        self.log("val_hausdorff", val_hausdorff, prog_bar=True, on_epoch=True, sync_dist=True)

        # Compute and log mean precision, recall, F1
        precision, recall, f1 = self.compute_metrics(val_outputs, decollate_batch(val_labels))
        self.log("val_mean_precision", precision, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mean_recall", recall, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mean_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_dice = self.dice_metric.aggregate().item()
        val_iou = self.jaccard_metric.aggregate().item()
        self.metric_values.append(val_dice)

        val_loss = self.trainer.logged_metrics["val_loss"].item()
        self.epoch_loss_values.append(val_loss)

        metric_batch = self.dice_metric_batch.aggregate()
        self.metric_values_tc.append(metric_batch[0].item())
        self.metric_values_wt.append(metric_batch[1].item())
        self.metric_values_et.append(metric_batch[2].item())

        # Log validation metrics
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mean_dice", val_dice, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mean_iou", val_iou, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_tc", metric_batch[0].item(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_wt", metric_batch[1].item(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_et", metric_batch[2].item(), prog_bar=True, on_epoch=True, sync_dist=True)

        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_metric_model_swinunetr_v2.pth")
            self.log("best_metric", self.best_metric, sync_dist=True, on_epoch=True)
            # Save best metrics for printing
            self.best_metric_tc = metric_batch[0].item()
            self.best_metric_wt = metric_batch[1].item()
            self.best_metric_et = metric_batch[2].item()
            self.best_metric_iou = val_iou
            # Try to get best Hausdorff from logs if available
            try:
                self.best_metric_hausdorff = self.trainer.logged_metrics["val_hausdorff"].item()
            except Exception:
                self.best_metric_hausdorff = float('nan')

        # Print best metrics after every validation epoch
        if hasattr(self.trainer, 'is_global_zero') and self.trainer.is_global_zero:
            print("\n=== Best Validation Metrics So Far ===")
            print(f"Best Mean Dice: {getattr(self, 'best_metric', float('nan')):.4f} at epoch {getattr(self, 'best_metric_epoch', 'N/A')}")
            print(f"Best TC Dice: {getattr(self, 'best_metric_tc', float('nan')):.4f}")
            print(f"Best WT Dice: {getattr(self, 'best_metric_wt', float('nan')):.4f}")
            print(f"Best ET Dice: {getattr(self, 'best_metric_et', float('nan')):.4f}")
            print(f"Best Mean IoU: {getattr(self, 'best_metric_iou', float('nan')):.4f}")
            print(f"Best Hausdorff: {getattr(self, 'best_metric_hausdorff', float('nan')):.4f}")
            print("======================================\n")

        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.jaccard_metric.reset()
        self.hausdorff_metric.reset()

    def on_train_end(self):
        print(f"Train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}, "
              f"tc: {self.metric_values_tc[-1]:.4f}, "
              f"wt: {self.metric_values_wt[-1]:.4f}, "
              f"et: {self.metric_values_et[-1]:.4f}.")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Warmup + Cosine Annealing
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return epoch / self.hparams.warmup_epochs
            else:
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.max_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate"
            }
        }