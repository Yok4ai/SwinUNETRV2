# pipeline.py - Training pipeline for Hybrid SwinUNETR
import torch
import pytorch_lightning as pl
import math
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from torch.optim import AdamW
from .hybrid_architecture import create_hybrid_swinunetr


class HybridBrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        max_epochs=50, 
        learning_rate=1e-4,
        # Hybrid model parameters
        efficiency_level="balanced",  # "light", "balanced", "performance"
        decoder_embedding_dim=128,
        use_segformer_decoder=True,
        # SwinUNETR backbone parameters
        feature_size=32,
        depths=(1, 1, 2, 1),
        num_heads=(2, 4, 8, 16),
        # Training parameters
        weight_decay=1e-5,
        warmup_epochs=5,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        decoder_dropout=0.0,
        # Inference parameters
        roi_size=(128, 128, 128),
        sw_batch_size=2,
        overlap=0.25,
        use_mixed_precision=True,
        norm_name="instance",
        use_checkpoint=True,
        use_v2=True,
        **kwargs  # Ignore extra parameters
    ):
        super().__init__()
        self.save_hyperparameters()
        
        print(f"🚀 Creating Hybrid SwinUNETR-SegFormer3D model")
        print(f"   Efficiency level: {efficiency_level}")
        print(f"   SegFormer decoder: {use_segformer_decoder}")
        print(f"   Decoder embedding dim: {decoder_embedding_dim}")
        
        # Create hybrid model
        self.model = create_hybrid_swinunetr(
            efficiency_level=efficiency_level,
            decoder_embedding_dim=decoder_embedding_dim,
            use_segformer_decoder=use_segformer_decoder,
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            use_v2=use_v2,
            spatial_dims=3,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            decoder_dropout=decoder_dropout,
        )
        
        # Loss and metrics
        self.loss_function = DiceLoss(
            smooth_nr=0, 
            smooth_dr=1e-5, 
            squared_pred=True, 
            to_onehot_y=False, 
            sigmoid=True
        )
        
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Training state
        self.best_metric = -1
        self.best_metric_epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.use_mixed_precision = use_mixed_precision

        # Metrics storage
        self.metric_values = []
        self.metric_values_tc = []
        self.metric_values_wt = []
        self.metric_values_et = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]

        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self(inputs)
                loss = self.loss_function(outputs, labels)
        else:
            outputs = self(inputs)
            loss = self.loss_function(outputs, labels)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # Compute metrics
        outputs_processed = [self.post_trans(i) for i in decollate_batch(outputs)]
        self.dice_metric(y_pred=outputs_processed, y=labels)
        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True, sync_dist=True)
        self.dice_metric.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        
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
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        val_outputs_processed = [self.post_trans(i) for i in decollate_batch(val_outputs)]    
        self.dice_metric(y_pred=val_outputs_processed, y=val_labels)
        self.dice_metric_batch(y_pred=val_outputs_processed, y=val_labels)
    
        val_dice = self.dice_metric.aggregate().item()
        self.log("val_mean_dice", val_dice, prog_bar=True, sync_dist=True)
    
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_dice = self.dice_metric.aggregate().item()
        self.metric_values.append(val_dice)

        # Individual class metrics
        metric_batch = self.dice_metric_batch.aggregate()
        self.metric_values_tc.append(metric_batch[0].item())
        self.metric_values_wt.append(metric_batch[1].item())
        self.metric_values_et.append(metric_batch[2].item())

        self.log("val_tc", metric_batch[0].item(), prog_bar=True, sync_dist=True)
        self.log("val_wt", metric_batch[1].item(), prog_bar=True, sync_dist=True)
        self.log("val_et", metric_batch[2].item(), prog_bar=True, sync_dist=True)
    
        # Save best model
        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_hybrid_swinunetr_segformer.pth")
            print(f"🎯 New best hybrid model saved! Dice: {self.best_metric:.4f}")
    
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def on_train_end(self):
        print(f"🏁 Hybrid training completed!")
        print(f"📊 Best metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
        print(f"🎯 Parameters: {self.model.count_parameters():,} ({self.model.count_parameters()/1e6:.2f}M)")
        print(f"🔗 Architecture: SwinUNETR V2 + SegFormer3D decoder")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
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
        }
        
        return [optimizer], [scheduler]

    def get_model_info(self):
        """Get detailed model information"""
        total_params = self.model.count_parameters()
        
        info = {
            "architecture": "Hybrid SwinUNETR-SegFormer3D",
            "efficiency_level": self.hparams.efficiency_level,
            "total_parameters": total_params,
            "parameters_mb": total_params / 1e6,
            "backbone": "SwinUNETR V2",
            "decoder": "SegFormer3D-style MLP" if self.hparams.use_segformer_decoder else "Standard",
            "decoder_embedding_dim": self.hparams.decoder_embedding_dim,
            "feature_size": self.hparams.feature_size,
            "use_v2_merging": self.hparams.use_v2,
            "parameter_reduction_vs_standard": f"{(1 - total_params / 62e6) * 100:.1f}%"
        }
        
        return info


# Configuration presets for easy usage
def get_hybrid_config(preset="balanced"):
    """Get predefined hybrid configurations"""
    
    configs = {
        "light": {
            "efficiency_level": "light",
            "decoder_embedding_dim": 96,
            "batch_size": 6,
            "accumulate_grad_batches": 2,
            "learning_rate": 8e-4,
            "sw_batch_size": 3,
        },
        "balanced": {
            "efficiency_level": "balanced", 
            "decoder_embedding_dim": 128,
            "batch_size": 4,
            "accumulate_grad_batches": 3,
            "learning_rate": 5e-4,
            "sw_batch_size": 2,
        },
        "performance": {
            "efficiency_level": "performance",
            "decoder_embedding_dim": 192,
            "batch_size": 3,
            "accumulate_grad_batches": 4,
            "learning_rate": 3e-4,
            "sw_batch_size": 2,
        }
    }
    
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}")
    
    return configs[preset]