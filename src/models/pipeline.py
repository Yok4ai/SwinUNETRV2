import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, TverskyLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete, RandFlipd
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
from .swinunetrplus import SwinUNETR
# from monai.networks.nets import SwinUNETR

class ModalityAttentionModule(nn.Module):
    """
    Modality Attention Module for better feature extraction across different MRI modalities.
    Learns importance weights for each modality channel.
    """
    def __init__(self, in_channels: int = 4, reduction_ratio: int = 2):
        super().__init__()
        self.in_channels = in_channels
        # Channel attention components
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        # Shared MLP
        hidden_channels = max(1, in_channels // reduction_ratio)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels)
        )
        # Spatial attention components
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channels, d, h, w = x.size()
        # Channel attention
        avg_pool = self.global_avg_pool(x).view(batch_size, channels)
        max_pool = self.global_max_pool(x).view(batch_size, channels)
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        channel_attention = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1, 1)
        x_channel = x * channel_attention
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_concat))
        x_refined = x_channel * spatial_attention
        return x_refined + x  # Residual connection

class BrainTumorSegmentation(pl.LightningModule):
    def load_state_dict(self, state_dict, strict=True):
        """Custom state dict loading with shape compatibility fixes"""
        # Fix focal_loss.class_weight shape mismatch
        if 'focal_loss.class_weight' in state_dict:
            weight = state_dict['focal_loss.class_weight']
            if weight.dim() == 4 and weight.shape == torch.Size([3, 1, 1, 1]):
                state_dict['focal_loss.class_weight'] = weight.squeeze()
        
        return super().load_state_dict(state_dict, strict=False)

    def __init__(self, train_loader, val_loader, max_epochs=100,
                 val_interval=1, learning_rate=1e-4, feature_size=48,
                 weight_decay=1e-5, warmup_epochs=10, roi_size=(96, 96, 96),
                 sw_batch_size=2, use_v2=True, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), downsample="mergingv2",
                 use_class_weights=True,
                 loss_type='dice',
                 tversky_alpha=0.5,
                 tversky_beta=0.5,
                 focal_gamma=2.0,
                 focal_alpha=None,
                 gdl_weight_type='square',
                 gdl_lambda=1.0,
                 hausdorff_alpha=2.0,
                 lambda_dice=1.0,
                 lambda_focal=1.0,
                 lambda_tversky=1.0,
                 lambda_hausdorff=1.0,
                 use_modality_attention=False,
                 overlap=0.7,
                 class_weights=(3.0, 1.0, 5.0),
                 threshold=0.5,
                 optimizer_betas=(0.9, 0.999),
                 optimizer_eps=1e-8,
                 # Adaptive loss scheduling parameters
                 use_adaptive_scheduling=False,
                 adaptive_schedule_type='linear',  # 'linear', 'exponential', 'cosine'
                 structure_epochs=30,  # Focus on structure learning
                 boundary_epochs=50,   # Focus on boundary refinement
                 schedule_start_epoch=10,  # When to start scheduling
                 min_loss_weight=0.1,     # Minimum weight for any loss
                 max_loss_weight=2.0,     # Maximum weight for any loss
                 # Warm restart parameters for local minima escape
                 use_warm_restarts=False,  # Enable cosine annealing with warm restarts
                 restart_period=20,        # Restart every N epochs
                 restart_mult=1,           # Multiplier for restart period
                 ):
        
        super().__init__()
        self.save_hyperparameters()
        self.use_modality_attention = use_modality_attention
        if self.use_modality_attention:
            self.modality_attention = ModalityAttentionModule(in_channels=4)
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
            class_weights = torch.tensor(list(self.hparams.class_weights))  # TC, WT, ET
        else:
            class_weights = None
            
        # Loss functions with class weighting
        self.loss_type = loss_type
        
        # Adaptive scheduling parameters
        self.use_adaptive_scheduling = use_adaptive_scheduling
        self.adaptive_schedule_type = adaptive_schedule_type
        self.structure_epochs = structure_epochs
        self.boundary_epochs = boundary_epochs
        self.schedule_start_epoch = schedule_start_epoch
        self.min_loss_weight = min_loss_weight
        self.max_loss_weight = max_loss_weight
        
        # Standard loss functions
        self.dice_loss = DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
            to_onehot_y=False, sigmoid=True, weight=class_weights
        )
        
        self.dicece_loss = DiceCELoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
            to_onehot_y=False, sigmoid=True, weight=class_weights,
            lambda_dice=self.hparams.lambda_dice, lambda_ce=1.0
        )
        
        self.dicefocal_loss = DiceFocalLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
            to_onehot_y=False, sigmoid=True, weight=class_weights,
            gamma=self.hparams.focal_gamma, alpha=self.hparams.focal_alpha,
            lambda_dice=self.hparams.lambda_dice, lambda_focal=self.hparams.lambda_focal
        )
        
        self.generalized_dice_loss = GeneralizedDiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True,
            w_type=self.hparams.gdl_weight_type
        )
        
        self.generalized_dice_focal_loss = GeneralizedDiceFocalLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True,
            w_type=self.hparams.gdl_weight_type, weight=class_weights,
            gamma=self.hparams.focal_gamma, 
            lambda_gdl=self.hparams.gdl_lambda, lambda_focal=self.hparams.lambda_focal
        )
        
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma, alpha=self.hparams.focal_alpha,
            weight=class_weights, reduction='mean', to_onehot_y=False
        )
        
        self.tversky_loss = TverskyLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True,
            alpha=self.hparams.tversky_alpha, beta=self.hparams.tversky_beta
        )
        
        self.hausdorff_loss = HausdorffDTLoss(
            alpha=self.hparams.hausdorff_alpha, include_background=False,
            to_onehot_y=False, sigmoid=True
        )
        
        # Standard Dice Loss Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.jaccard_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=self.hparams.threshold)])
        
        self.best_metric = -1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.overlap = overlap

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
        if self.use_modality_attention:
            x = self.modality_attention(x)
        return self.model(x)

    def compute_loss(self, outputs, labels):
        """Compute loss based on loss_type configuration"""
        if self.loss_type == 'dice':
            return self.dice_loss(outputs, labels)
        elif self.loss_type == 'dicece':
            return self.dicece_loss(outputs, labels)
        elif self.loss_type == 'dicefocal':
            return self.dicefocal_loss(outputs, labels)
        elif self.loss_type == 'generalized_dice':
            return self.generalized_dice_loss(outputs, labels)
        elif self.loss_type == 'generalized_dice_focal':
            return self.generalized_dice_focal_loss(outputs, labels)
        elif self.loss_type == 'focal':
            return self.focal_loss(outputs, labels)
        elif self.loss_type == 'tversky':
            return self.tversky_loss(outputs, labels)
        elif self.loss_type == 'hausdorff':
            return self.hausdorff_loss(outputs, labels)
        elif self.loss_type == 'hybrid_gdl_focal_tversky':
            # Generalized Dice + Focal + Tversky (controlled mix)
            gdl_loss = self.generalized_dice_loss(outputs, labels)
            focal_loss = self.focal_loss(outputs, labels)
            tversky_loss = self.tversky_loss(outputs, labels)
            total_loss = (
                self.hparams.gdl_lambda * gdl_loss + 
                self.hparams.lambda_focal * focal_loss + 
                self.hparams.lambda_tversky * tversky_loss
            )
            return total_loss
        elif self.loss_type == 'hybrid_dice_hausdorff':
            # Dice + Hausdorff
            dice_loss = self.dice_loss(outputs, labels)
            hausdorff_loss = self.hausdorff_loss(outputs, labels)
            total_loss = (
                self.hparams.lambda_dice * dice_loss + 
                self.hparams.lambda_hausdorff * hausdorff_loss
            )
            return total_loss
        elif self.loss_type == 'adaptive_structure_boundary':
            # Adaptive loss that schedules between structure (Dice) and boundary (Focal) learning
            dice_loss = self.dice_loss(outputs, labels)
            focal_loss = self.focal_loss(outputs, labels)
            
            dice_weight, focal_weight = self._get_adaptive_weights()
            total_loss = dice_weight * dice_loss + focal_weight * focal_loss
            return total_loss
        elif self.loss_type == 'adaptive_progressive_hybrid':
            # Progressive loss that starts with simple Dice, adds complexity over time
            dice_loss = self.dice_loss(outputs, labels)
            focal_loss = self.focal_loss(outputs, labels)
            tversky_loss = self.tversky_loss(outputs, labels)
            
            dice_weight, focal_weight, tversky_weight = self._get_progressive_weights()
            total_loss = (
                dice_weight * dice_loss + 
                focal_weight * focal_loss + 
                tversky_weight * tversky_loss
            )
            return total_loss
        elif self.loss_type == 'adaptive_complexity_cascade':
            # Cascading complexity: Dice -> DiceCE -> DiceFocal + Hausdorff
            dice_loss = self.dice_loss(outputs, labels)
            dicece_loss = self.dicece_loss(outputs, labels)
            dicefocal_loss = self.dicefocal_loss(outputs, labels)
            hausdorff_loss = self.hausdorff_loss(outputs, labels)
            
            weights = self._get_cascade_weights()
            total_loss = (
                weights['dice'] * dice_loss +
                weights['dicece'] * dicece_loss +
                weights['dicefocal'] * dicefocal_loss +
                weights['hausdorff'] * hausdorff_loss
            )
            return total_loss
        elif self.loss_type == 'adaptive_dynamic_hybrid':
            # Dynamic hybrid that adapts weights based on validation performance
            gdl_loss = self.generalized_dice_loss(outputs, labels)
            focal_loss = self.focal_loss(outputs, labels)
            tversky_loss = self.tversky_loss(outputs, labels)
            hausdorff_loss = self.hausdorff_loss(outputs, labels)
            
            weights = self._get_dynamic_weights()
            total_loss = (
                weights['gdl'] * gdl_loss +
                weights['focal'] * focal_loss +
                weights['tversky'] * tversky_loss +
                weights['hausdorff'] * hausdorff_loss
            )
            return total_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _get_adaptive_weights(self):
        """Get adaptive weights for structure-boundary scheduling"""
        if not self.use_adaptive_scheduling or self.current_epoch < self.schedule_start_epoch:
            return 1.0, 1.0
        
        epoch = self.current_epoch - self.schedule_start_epoch
        max_epoch = self.hparams.max_epochs - self.schedule_start_epoch
        
        if self.adaptive_schedule_type == 'linear':
            # Linear transition from structure focus to boundary focus
            progress = min(epoch / max_epoch, 1.0)
            dice_weight = self.max_loss_weight * (1.0 - progress) + self.min_loss_weight * progress
            focal_weight = self.min_loss_weight * (1.0 - progress) + self.max_loss_weight * progress
        elif self.adaptive_schedule_type == 'exponential':
            # Exponential decay for structure, exponential growth for boundary
            progress = min(epoch / max_epoch, 1.0)
            dice_weight = self.max_loss_weight * (0.5 ** (progress * 3))
            focal_weight = self.min_loss_weight + (self.max_loss_weight - self.min_loss_weight) * (1 - 0.5 ** (progress * 3))
        elif self.adaptive_schedule_type == 'cosine':
            # Cosine scheduling for smooth transitions
            progress = min(epoch / max_epoch, 1.0)
            dice_weight = self.min_loss_weight + (self.max_loss_weight - self.min_loss_weight) * 0.5 * (1 + math.cos(math.pi * progress))
            focal_weight = self.min_loss_weight + (self.max_loss_weight - self.min_loss_weight) * 0.5 * (1 - math.cos(math.pi * progress))
        else:
            dice_weight, focal_weight = 1.0, 1.0
            
        return dice_weight, focal_weight
    
    def _get_progressive_weights(self):
            """Get progressive weights that add complexity over time"""
            if not self.use_adaptive_scheduling or self.current_epoch < self.schedule_start_epoch:
                return 1.0, 0.0, 0.0
            
            epoch = self.current_epoch - self.schedule_start_epoch
            
            # Phase 1: Structure learning (Dice dominant)
            if epoch < self.structure_epochs:
                dice_weight = self.max_loss_weight
                focal_weight = self.min_loss_weight
                tversky_weight = 0.0
            # Phase 2: Add boundary refinement (Focal)
            elif epoch < self.boundary_epochs:
                progress = (epoch - self.structure_epochs) / (self.boundary_epochs - self.structure_epochs)
                dice_weight = self.max_loss_weight * (1.0 - 0.3 * progress)
                focal_weight = self.min_loss_weight + (self.max_loss_weight - self.min_loss_weight) * progress
                tversky_weight = 0.0
            # Phase 3: Add precision/recall balance (Tversky)
            else:
                progress = min((epoch - self.boundary_epochs) / 15, 1.0)  # 15 epochs to ramp up
                dice_weight = self.max_loss_weight * 0.8   # Maintain strong dice
                focal_weight = self.max_loss_weight * 0.7  # Keep focal strong
                tversky_weight = self.min_loss_weight + (self.max_loss_weight * 0.4 - self.min_loss_weight) * progress  # Moderate tversky
                
            return dice_weight, focal_weight, tversky_weight
    
    def _get_cascade_weights(self):
        """Get cascading complexity weights"""
        if not self.use_adaptive_scheduling or self.current_epoch < self.schedule_start_epoch:
            return {'dice': 1.0, 'dicece': 0.0, 'dicefocal': 0.0, 'hausdorff': 0.0}
        
        epoch = self.current_epoch - self.schedule_start_epoch
        max_epoch = self.hparams.max_epochs - self.schedule_start_epoch
        
        # Stage 1: Pure Dice (0-25%)
        # Stage 2: Dice + DiceCE (25-50%)
        # Stage 3: Dice + DiceCE + DiceFocal (50-75%)
        # Stage 4: All losses (75-100%)
        
        progress = min(epoch / max_epoch, 1.0)
        
        if progress < 0.25:
            weights = {'dice': self.max_loss_weight, 'dicece': 0.0, 'dicefocal': 0.0, 'hausdorff': 0.0}
        elif progress < 0.5:
            stage_progress = (progress - 0.25) / 0.25
            weights = {
                'dice': self.max_loss_weight * (1.0 - 0.3 * stage_progress),
                'dicece': self.min_loss_weight + (self.max_loss_weight * 0.7 - self.min_loss_weight) * stage_progress,
                'dicefocal': 0.0,
                'hausdorff': 0.0
            }
        elif progress < 0.75:
            stage_progress = (progress - 0.5) / 0.25
            weights = {
                'dice': self.max_loss_weight * 0.7,
                'dicece': self.max_loss_weight * 0.7,
                'dicefocal': self.min_loss_weight + (self.max_loss_weight * 0.8 - self.min_loss_weight) * stage_progress,
                'hausdorff': 0.0
            }
        else:
            stage_progress = (progress - 0.75) / 0.25
            weights = {
                'dice': self.max_loss_weight * 0.7,
                'dicece': self.max_loss_weight * 0.7,
                'dicefocal': self.max_loss_weight * 0.8,
                'hausdorff': self.min_loss_weight + (self.max_loss_weight * 0.5 - self.min_loss_weight) * stage_progress
            }
            
        return weights
    
    def _get_dynamic_weights(self):
        """Get dynamic weights based on current performance"""
        if not self.use_adaptive_scheduling or self.current_epoch < self.schedule_start_epoch:
            return {'gdl': 1.0, 'focal': 0.5, 'tversky': 0.5, 'hausdorff': 0.0}
        
        # Get recent validation dice if available
        recent_dice = getattr(self, 'best_metric', 0.5)
        
        # Adaptive strategy based on performance
        if recent_dice < 0.7:  # Poor performance - focus on structure
            weights = {
                'gdl': self.max_loss_weight,
                'focal': self.min_loss_weight,
                'tversky': self.min_loss_weight,
                'hausdorff': 0.0
            }
        elif recent_dice < 0.85:  # Moderate performance - balanced approach
            weights = {
                'gdl': self.max_loss_weight * 0.8,
                'focal': self.max_loss_weight * 0.6,
                'tversky': self.max_loss_weight * 0.6,
                'hausdorff': self.min_loss_weight
            }
        else:  # Good performance - focus on fine details
            weights = {
                'gdl': self.max_loss_weight * 0.6,
                'focal': self.max_loss_weight,
                'tversky': self.max_loss_weight * 0.8,
                'hausdorff': self.max_loss_weight * 0.7
            }
            
        return weights

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

        # Standard sliding window inference
        val_outputs = sliding_window_inference(
            val_inputs, roi_size=self.hparams.roi_size, sw_batch_size=1, 
            predictor=self.model, overlap=self.overlap  # Tunable overlap
        )
        
        # Compute loss with hybrid approach
        val_loss = self.compute_loss(val_outputs, val_labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]    

        # Log images to wandb (only for the first batch of each epoch)
        if batch_idx == 0 and self.logger is not None and hasattr(self.logger, "experiment"):
            # Take the first image in the batch
            img = val_inputs[0, 0].detach().cpu().numpy()  # First channel, first image
            pred = val_outputs[0][0].detach().cpu().numpy()
            label = val_labels[0, 0].detach().cpu().numpy()
            # Normalize for visualization
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Pick a middle slice for 3D volumes
            slice_idx = img.shape[-1] // 2
            self.logger.experiment.log({
                "val_image": wandb.Image(img[..., slice_idx], caption="Input"),
                "val_pred": wandb.Image(pred[..., slice_idx], caption="Prediction"),
                "val_label": wandb.Image(label[..., slice_idx], caption="Label"),
                "global_step": self.global_step
            })

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
            betas=self.hparams.optimizer_betas,
            eps=self.hparams.optimizer_eps
        )
        
        if self.hparams.use_warm_restarts:
            # Cosine Annealing with Warm Restarts (helps escape local minima)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.restart_period,  # Initial restart period
                T_mult=self.hparams.restart_mult,  # Period multiplier after restart
                eta_min=self.hparams.learning_rate * 0.01,  # Minimum LR (1% of initial)
                last_epoch=-1
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "learning_rate"
                }
            }
        else:
            # Original Warmup + Cosine Annealing
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