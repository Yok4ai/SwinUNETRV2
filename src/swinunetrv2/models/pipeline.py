# pipeline.py
import torch
import pytorch_lightning as pl
import math
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.networks.nets import SwinUNETR
from torch.optim import AdamW
import torch.nn as nn
from .architecture import ParameterEfficientSwinUNETR


class BrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        max_epochs=50, 
        val_interval=1, 
        learning_rate=1e-4,
        feature_size=32,     # MONAI SwinUNETR parameter
        depths=(2, 2, 2, 2), # MONAI SwinUNETR parameter
        num_heads=(3, 6, 12, 24), # MONAI SwinUNETR parameter
        weight_decay=1e-5,
        warmup_epochs=5,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        roi_size=(128, 128, 128),
        sw_batch_size=2,
        overlap=0.25,
        use_mixed_precision=True,
        norm_name="instance",
        use_checkpoint=True,
        use_v2=True,
        # Legacy parameters (ignored but kept for compatibility)
        img_size=128,
        embed_dim=None,
        window_size=None,
        mlp_ratio=None,
        decoder_embed_dim=None,
        patch_size=None,
        **kwargs  # Catch any other legacy parameters
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Warn about ignored parameters
        ignored_params = []
        if embed_dim is not None: ignored_params.append(f"embed_dim={embed_dim}")
        if window_size is not None: ignored_params.append(f"window_size={window_size}")
        if mlp_ratio is not None: ignored_params.append(f"mlp_ratio={mlp_ratio}")
        if decoder_embed_dim is not None: ignored_params.append(f"decoder_embed_dim={decoder_embed_dim}")
        if patch_size is not None: ignored_params.append(f"patch_size={patch_size}")
        
        if ignored_params:
            print(f"⚠️  Ignoring custom implementation parameters: {', '.join(ignored_params)}")
            print("   Using MONAI SwinUNETR standard parameters instead")
        
        # Use MONAI's proven SwinUNETR
        self.model = ParameterEfficientSwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            use_v2=use_v2,
            spatial_dims=3,
            depths=depths,
            num_heads=num_heads,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            downsample="mergingv2"
        )
        
        # Print model size
        total_params = self.model.count_parameters()
        print(f"🚀 MONAI SwinUNETR initialized with {total_params:,} parameters ({total_params/1e6:.2f}M)")
        
        # Use proven loss configuration
        self.loss_function = DiceLoss(
            smooth_nr=0, 
            smooth_dr=1e-5, 
            squared_pred=True, 
            to_onehot_y=False, 
            sigmoid=True
        )
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Training parameters
        self.best_metric = -1
        self.best_metric_epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.use_mixed_precision = use_mixed_precision

        # Metrics storage
        self._init_metrics_storage()

    def _init_metrics_storage(self):
        """Initialize metric storage lists"""
        self.avg_train_loss_values = []
        self.train_loss_values = []
        self.train_metric_values = []
        self.train_metric_values_tc = []
        self.train_metric_values_wt = []
        self.train_metric_values_et = []

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

        # Forward pass with optional mixed precision
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self(inputs)
                loss = self.loss_function(outputs, labels)
        else:
            outputs = self(inputs)
            loss = self.loss_function(outputs, labels)
        
        # Log training loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # Post-process outputs for metrics
        outputs_processed = [self.post_trans(i) for i in decollate_batch(outputs)]
        
        # Compute metrics
        self.dice_metric(y_pred=outputs_processed, y=labels)
        self.dice_metric_batch(y_pred=outputs_processed, y=labels)

        # Log training metrics
        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True, sync_dist=True)

        # Store metrics
        self.train_metric_values.append(train_dice)

        # Individual class metrics
        metric_batch = self.dice_metric_batch.aggregate()
        self.train_metric_values_tc.append(metric_batch[0].item())
        self.train_metric_values_wt.append(metric_batch[1].item())
        self.train_metric_values_et.append(metric_batch[2].item())

        # Log individual metrics
        self.log("train_tc", metric_batch[0].item(), prog_bar=True, sync_dist=True)
        self.log("train_wt", metric_batch[1].item(), prog_bar=True, sync_dist=True)
        self.log("train_et", metric_batch[2].item(), prog_bar=True, sync_dist=True)

        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        
        # Use sliding window inference for validation
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    val_inputs, 
                    roi_size=self.roi_size, 
                    sw_batch_size=self.sw_batch_size, 
                    predictor=self.model, 
                    overlap=self.overlap
                )
                val_loss = self.loss_function(val_outputs, val_labels)
        else:
            val_outputs = sliding_window_inference(
                val_inputs, 
                roi_size=self.roi_size, 
                sw_batch_size=self.sw_batch_size, 
                predictor=self.model, 
                overlap=self.overlap
            )
            val_loss = self.loss_function(val_outputs, val_labels)
        
        # Log validation loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        # Post-process for metrics
        val_outputs_processed = [self.post_trans(i) for i in decollate_batch(val_outputs)]    
        
        # Compute metrics
        self.dice_metric(y_pred=val_outputs_processed, y=val_labels)
        self.dice_metric_batch(y_pred=val_outputs_processed, y=val_labels)
    
        # Log validation dice
        val_dice = self.dice_metric.aggregate().item()
        self.log("val_mean_dice", val_dice, prog_bar=True, sync_dist=True)
    
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        """Handle validation epoch end"""
        # Aggregate metrics
        val_dice = self.dice_metric.aggregate().item()
        self.metric_values.append(val_dice)

        # Store validation loss
        if "val_loss" in self.trainer.logged_metrics:
            val_loss = self.trainer.logged_metrics["val_loss"].item()
            self.epoch_loss_values.append(val_loss)

            # Calculate average validation loss
            avg_val_loss = sum(self.epoch_loss_values) / len(self.epoch_loss_values)
            self.log("avg_val_loss", avg_val_loss, prog_bar=True)
            self.avg_val_loss_values.append(avg_val_loss)

        # Individual class metrics
        metric_batch = self.dice_metric_batch.aggregate()
        self.metric_values_tc.append(metric_batch[0].item())
        self.metric_values_wt.append(metric_batch[1].item())
        self.metric_values_et.append(metric_batch[2].item())

        # Log individual validation metrics
        self.log("val_tc", metric_batch[0].item(), prog_bar=True, sync_dist=True)
        self.log("val_wt", metric_batch[1].item(), prog_bar=True, sync_dist=True)
        self.log("val_et", metric_batch[2].item(), prog_bar=True, sync_dist=True)
    
        # Save best model
        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_metric_model_monai_swinunetr.pth")
            self.log("best_metric", self.best_metric)
            print(f"🎯 New best model saved! Dice: {self.best_metric:.4f}")
    
        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def on_train_epoch_end(self):
        """Handle training epoch end"""
        if "train_loss" in self.trainer.logged_metrics:
            train_loss = self.trainer.logged_metrics["train_loss"].item()
            self.train_loss_values.append(train_loss)
            
            # Calculate average training loss
            avg_train_loss = sum(self.train_loss_values) / len(self.train_loss_values)
            self.log("avg_train_loss", avg_train_loss, prog_bar=True)
            self.avg_train_loss_values.append(avg_train_loss)

    def on_train_end(self):
        """Print final results"""
        print(f"🏁 Training completed!")
        print(f"📊 Best metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
        if len(self.metric_values_tc) > 0:
            print(f"🎯 Final metrics - TC: {self.metric_values_tc[-1]:.4f}, "
                  f"WT: {self.metric_values_wt[-1]:.4f}, "
                  f"ET: {self.metric_values_et[-1]:.4f}")

    def configure_optimizers(self):
        """Configure optimizer with proven settings for SwinUNETR"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        total_steps = len(self.train_loader) * self.hparams.max_epochs
        warmup_steps = len(self.train_loader) * self.hparams.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
            'name': 'learning_rate'
        }
        
        return [optimizer], [scheduler]