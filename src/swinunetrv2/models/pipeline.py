# pipeline.py - Updated with Ultra-Efficient SwinUNETR
import torch
import pytorch_lightning as pl
import math
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.optim import AdamW
import torch.nn as nn
from .architecture import create_efficient_swinunetr, UltraEfficientSwinUNETR


class BrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        max_epochs=50, 
        val_interval=1, 
        learning_rate=1e-4,
        # New efficiency parameters
        efficiency_level="balanced",  # "ultra", "high", "balanced", "performance"
        feature_size=24,     # Override for custom config
        depths=(1, 1, 2, 1), # Override for custom config
        num_heads=(2, 4, 8, 16), # Override for custom config
        decoder_channels=(96, 48, 24, 12),  # New parameter for lightweight decoder
        use_segformer_style=False,  # Use SegFormer3D-style extreme efficiency
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
            print("   Using Ultra-Efficient SwinUNETR parameters instead")
        
        # Create ultra-efficient model
        if use_segformer_style:
            print("🚀 Using SegFormer3D-style Ultra-Efficient SwinUNETR")
            from .architecture import SegFormerStyleSwinUNETR
            self.model = SegFormerStyleSwinUNETR(
                in_channels=4,
                out_channels=3,
                use_checkpoint=use_checkpoint,
                use_v2=use_v2,
                spatial_dims=3,
                norm_name=norm_name,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
            )
        else:
            # Use factory function for different efficiency levels
            print(f"🚀 Using {efficiency_level.upper()} efficiency Ultra-Efficient SwinUNETR")
            
            # Custom configuration if specific parameters provided
            custom_config = {}
            if feature_size != 24 or depths != (1, 1, 2, 1) or num_heads != (2, 4, 8, 16):
                custom_config.update({
                    'feature_size': feature_size,
                    'depths': depths,
                    'num_heads': num_heads,
                    'decoder_channels': decoder_channels,
                })
                print("🔧 Using custom configuration parameters")
            
            self.model = create_efficient_swinunetr(
                efficiency_level=efficiency_level,
                in_channels=4,
                out_channels=3,
                use_checkpoint=use_checkpoint,
                use_v2=use_v2,
                spatial_dims=3,
                norm_name=norm_name,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
                **custom_config
            )
        
        # Print efficiency comparison
        total_params = self.model.count_parameters()
        self._print_efficiency_comparison(total_params, efficiency_level)
        
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

    def _print_efficiency_comparison(self, total_params, efficiency_level):
        """Print efficiency comparison with other models"""
        print(f"\n📊 Model Efficiency Comparison:")
        print(f"   🎯 Current model: {total_params:,} parameters ({total_params/1e6:.2f}M)")
        
        # Reference comparisons
        comparisons = {
            "SegFormer3D (reference)": "~8-15M",
            "Standard MONAI SwinUNETR": "~62M", 
            "UNet3D": "~30-40M",
            "nnU-Net": "~31M"
        }
        
        print(f"   📈 Efficiency level: {efficiency_level.upper()}")
        print(f"   📋 Reference comparisons:")
        for model, params in comparisons.items():
            print(f"      • {model}: {params}")
        
        # Calculate efficiency vs standard SwinUNETR
        standard_params = 62e6
        efficiency_gain = (1 - total_params / standard_params) * 100
        print(f"   🎉 Parameter reduction: {efficiency_gain:.1f}% vs standard SwinUNETR")
        
        # Memory estimation
        memory_estimate = total_params * 4 / 1e9  # Rough estimate in GB
        print(f"   💾 Estimated model memory: ~{memory_estimate:.2f} GB")

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
            torch.save(self.model.state_dict(), "best_metric_model_ultra_efficient_swinunetr.pth")
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
        print(f"🏁 Ultra-Efficient SwinUNETR training completed!")
        print(f"📊 Best metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
        print(f"🎯 Final parameter count: {self.model.count_parameters():,} ({self.model.count_parameters()/1e6:.2f}M)")
        
        if len(self.metric_values_tc) > 0:
            print(f"🎯 Final metrics - TC: {self.metric_values_tc[-1]:.4f}, "
                  f"WT: {self.metric_values_wt[-1]:.4f}, "
                  f"ET: {self.metric_values_et[-1]:.4f}")
        
        # Calculate efficiency metrics
        standard_params = 62e6  # Standard MONAI SwinUNETR
        efficiency_ratio = self.model.count_parameters() / standard_params
        print(f"📈 Efficiency: {efficiency_ratio:.2f}x parameters vs standard SwinUNETR")
        print(f"🚀 Parameter reduction: {(1-efficiency_ratio)*100:.1f}%")

    def configure_optimizers(self):
        """Configure optimizer with proven settings for Ultra-Efficient SwinUNETR"""
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

    def get_efficiency_summary(self):
        """Get detailed efficiency summary"""
        total_params = self.model.count_parameters()
        
        summary = {
            "model_type": "Ultra-Efficient SwinUNETR",
            "efficiency_level": getattr(self.hparams, 'efficiency_level', 'custom'),
            "total_parameters": total_params,
            "parameters_mb": total_params / 1e6,
            "feature_size": getattr(self.hparams, 'feature_size', 'N/A'),
            "depths": getattr(self.hparams, 'depths', 'N/A'),
            "num_heads": getattr(self.hparams, 'num_heads', 'N/A'),
            "decoder_channels": getattr(self.hparams, 'decoder_channels', 'N/A'),
            "segformer_style": getattr(self.hparams, 'use_segformer_style', False),
            "parameter_reduction_vs_standard": f"{(1 - total_params / 62e6) * 100:.1f}%",
            "estimated_memory_gb": f"{total_params * 4 / 1e9:.2f}"
        }
        
        return summary


# Enhanced efficiency configurations for easy selection
def get_ultra_lightweight_config():
    """SegFormer3D-style ultra lightweight - ~5-8M parameters"""
    return {
        "efficiency_level": "ultra",
        "use_segformer_style": True,
        "batch_size": 8,
        "accumulate_grad_batches": 2,  # effective batch = 16
        "learning_rate": 1e-3,  # Higher LR for smaller model
        "sw_batch_size": 4,  # Can afford larger sliding window batch
    }

def get_high_efficiency_config():
    """High efficiency - ~10-15M parameters"""
    return {
        "efficiency_level": "high",
        "batch_size": 6,
        "accumulate_grad_batches": 2,  # effective batch = 12
        "learning_rate": 8e-4,
        "sw_batch_size": 3,
    }

def get_balanced_efficiency_config():
    """Balanced efficiency - ~15-25M parameters"""
    return {
        "efficiency_level": "balanced",
        "batch_size": 4,
        "accumulate_grad_batches": 3,  # effective batch = 12
        "learning_rate": 5e-4,
        "sw_batch_size": 2,
    }

def get_performance_efficiency_config():
    """Performance focused - ~25-35M parameters"""
    return {
        "efficiency_level": "performance",
        "batch_size": 3,
        "accumulate_grad_batches": 4,  # effective batch = 12
        "learning_rate": 3e-4,
        "sw_batch_size": 2,
    }

def get_custom_segformer_config():
    """Custom SegFormer3D-inspired configuration"""
    return {
        "efficiency_level": "balanced",
        "feature_size": 16,  # Like SegFormer3D
        "depths": (1, 1, 1, 1),  # Minimal depths
        "num_heads": (1, 2, 4, 8),  # Small attention heads
        "decoder_channels": (64, 32, 16, 8),  # Tiny decoder
        "batch_size": 6,
        "accumulate_grad_batches": 2,
        "learning_rate": 1e-3,
    }