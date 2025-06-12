#pipeline.py
import torch
import pytorch_lightning as pl
import math
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from torch.optim import AdamW
from .architecture import LightweightSwinUNETR

class BrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        max_epochs=50, 
        val_interval=1, 
        learning_rate=5e-4,
        img_size=128,
        feature_size=48,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        decoder_embed_dim=256,
        patch_size=4,
        weight_decay=1e-5,
        warmup_epochs=5,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        roi_size=(128, 128, 128),
        sw_batch_size=2,
        overlap=0.25
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = LightweightSwinUNETR(
            in_channels=4,
            out_channels=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            decoder_embed_dim=decoder_embed_dim,
            patch_size=patch_size,
            dropout=drop_rate
        )
        
        # Print model size
        total_params = self.model.count_parameters()
        print(f"🚀 Model initialized with {total_params:,} parameters ({total_params/1e6:.2f}M)")
        
        # FIXED: Use class weights for DiceCE loss - properly handle device placement
        self.register_buffer('class_weights', torch.tensor([1.0, 2.0, 4.0]))
        
        # FIXED: Separate Dice and CE losses for better control and per-class weighting
        self.dice_loss = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=False,
            reduction="mean",
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            batch=False
        )
        
        # FIXED: Use DiceCELoss with proper configuration for multi-class segmentation
        self.dice_ce_loss = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=False,
            reduction="mean",
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            batch=False,
            weight=None,  # Will apply class weights manually
            lambda_dice=1.0,
            lambda_ce=1.0
        )
        
        # FIXED: Per-class dice metrics for detailed monitoring
        self.dice_metric_mean = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
        
        # FIXED: Individual class metrics for better tracking
        self.dice_metric_tc = DiceMetric(include_background=False, reduction="mean_batch")
        self.dice_metric_wt = DiceMetric(include_background=False, reduction="mean_batch") 
        self.dice_metric_et = DiceMetric(include_background=False, reduction="mean_batch")
        
        self.best_metric = -1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap

        # Training metrics storage
        self.train_metrics = {
            'loss': [], 'dice_mean': [], 'dice_tc': [], 'dice_wt': [], 'dice_et': []
        }
        self.val_metrics = {
            'loss': [], 'dice_mean': [], 'dice_tc': [], 'dice_wt': [], 'dice_et': []
        }

    def forward(self, x):
        return self.model(x)

    def compute_loss_with_class_weights(self, outputs, labels):
        """Compute weighted loss considering class imbalance"""
        # FIXED: Ensure labels are in correct format for MONAI losses
        # Convert multi-channel one-hot labels to single-channel class indices
        if labels.shape[1] > 1:  # If labels are one-hot encoded
            labels = torch.argmax(labels, dim=1, keepdim=True)
        
        # Ensure labels are the right type
        labels = labels.long()
        
        # Base DiceCE loss with proper label format
        base_loss = self.dice_ce_loss(outputs, labels)
        
        # FIXED: Apply class weights to the loss
        # Convert outputs to probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Convert labels to one-hot for class weight computation
        labels_onehot = torch.zeros_like(outputs)
        labels_onehot.scatter_(1, labels, 1)
            
        # Compute per-class weights based on presence in batch
        class_presence = labels_onehot.sum(dim=[0, 2, 3, 4])  # Sum over batch and spatial dims
        
        # Apply class weights where classes are present
        weighted_loss = base_loss
        for i, (presence, weight) in enumerate(zip(class_presence, self.class_weights)):
            if presence > 0:  # Only weight if class is present
                class_mask = labels_onehot[:, i:i+1]  # Get mask for this class
                # Use dice loss without to_onehot_y since we're providing one-hot already
                class_dice_loss = DiceLoss(
                    include_background=False,
                    to_onehot_y=False,  # Already one-hot
                    softmax=False,      # Already softmax
                    reduction="mean"
                )
                class_loss = class_dice_loss(probs[:, i:i+1], class_mask[:, i:i+1])
                weighted_loss += (weight - 1.0) * class_loss * 0.1  # Small additional weighting
        
        return weighted_loss

    def compute_metrics(self, outputs, labels, prefix=""):
        """Compute all dice metrics and log them"""
        # Convert outputs to probabilities and then to predictions
        outputs_softmax = torch.softmax(outputs, dim=1)
        outputs_pred = torch.argmax(outputs_softmax, dim=1, keepdim=True)
        
        # Convert predictions to one-hot
        outputs_onehot = torch.zeros_like(outputs)
        outputs_onehot.scatter_(1, outputs_pred, 1)
        
        # FIXED: Ensure labels are in correct format (single channel with class indices)
        if labels.shape[1] > 1:  # If labels are one-hot encoded
            labels_single = torch.argmax(labels, dim=1, keepdim=True)
        else:
            labels_single = labels
        
        # Ensure labels are long type
        labels_single = labels_single.long()
        
        # Convert labels to one-hot for metric computation
        labels_onehot = torch.zeros_like(outputs)
        labels_onehot.scatter_(1, labels_single, 1)
        
        # Compute overall mean dice
        self.dice_metric_mean(y_pred=outputs_onehot, y=labels_onehot)
        mean_dice = self.dice_metric_mean.aggregate().item()
        
        # Compute per-class dice
        self.dice_metric_batch(y_pred=outputs_onehot, y=labels_onehot)
        batch_dice = self.dice_metric_batch.aggregate()
        
        # Extract individual class scores (TC, WT, ET)
        dice_scores = {
            'mean': mean_dice,
            'tc': batch_dice[0].item() if len(batch_dice) > 0 else 0.0,
            'wt': batch_dice[1].item() if len(batch_dice) > 1 else 0.0,
            'et': batch_dice[2].item() if len(batch_dice) > 2 else 0.0
        }
        
        # FIXED: Log ALL metrics to progress bar
        self.log(f"{prefix}dice_mean", dice_scores['mean'], prog_bar=True, sync_dist=True)
        self.log(f"{prefix}dice_tc", dice_scores['tc'], prog_bar=True, sync_dist=True)
        self.log(f"{prefix}dice_wt", dice_scores['wt'], prog_bar=True, sync_dist=True)
        self.log(f"{prefix}dice_et", dice_scores['et'], prog_bar=True, sync_dist=True)
        
        # Reset metrics
        self.dice_metric_mean.reset()
        self.dice_metric_batch.reset()
        
        return dice_scores

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        outputs = self(inputs)
        
        # FIXED: Compute weighted loss
        loss = self.compute_loss_with_class_weights(outputs, labels)
        
        # Log training loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        # FIXED: Compute and log all training metrics
        dice_scores = self.compute_metrics(outputs, labels, prefix="train_")
        
        # Store metrics for epoch-level tracking
        self.train_metrics['loss'].append(loss.item())
        self.train_metrics['dice_mean'].append(dice_scores['mean'])
        self.train_metrics['dice_tc'].append(dice_scores['tc'])
        self.train_metrics['dice_wt'].append(dice_scores['wt'])
        self.train_metrics['dice_et'].append(dice_scores['et'])

        return loss

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        
        # Use sliding window inference for validation
        val_outputs = sliding_window_inference(
            val_inputs, 
            roi_size=self.roi_size, 
            sw_batch_size=self.sw_batch_size, 
            predictor=self.model,
            overlap=self.overlap
        )
        
        # FIXED: Compute weighted validation loss
        val_loss = self.compute_loss_with_class_weights(val_outputs, val_labels)
        
        # Log validation loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        # FIXED: Compute and log all validation metrics
        dice_scores = self.compute_metrics(val_outputs, val_labels, prefix="val_")
        
        return {
            "val_loss": val_loss, 
            "val_dice_mean": dice_scores['mean'],
            "val_dice_tc": dice_scores['tc'],
            "val_dice_wt": dice_scores['wt'],
            "val_dice_et": dice_scores['et']
        }

    def on_validation_epoch_end(self):
        # Get current validation metrics from logged values
        current_metrics = {
            'mean': self.trainer.logged_metrics.get("val_dice_mean", 0.0).item(),
            'tc': self.trainer.logged_metrics.get("val_dice_tc", 0.0).item(),
            'wt': self.trainer.logged_metrics.get("val_dice_wt", 0.0).item(),
            'et': self.trainer.logged_metrics.get("val_dice_et", 0.0).item()
        }
        
        # Store in validation metrics
        self.val_metrics['dice_mean'].append(current_metrics['mean'])
        self.val_metrics['dice_tc'].append(current_metrics['tc'])
        self.val_metrics['dice_wt'].append(current_metrics['wt'])
        self.val_metrics['dice_et'].append(current_metrics['et'])
        
        if "val_loss" in self.trainer.logged_metrics:
            self.val_metrics['loss'].append(self.trainer.logged_metrics["val_loss"].item())
        
        # FIXED: Check for best metric based on mean dice
        current_dice = current_metrics['mean']
        if current_dice > self.best_metric:
            self.best_metric = current_dice
            self.best_metric_epoch = self.current_epoch
            
            # Save best model
            torch.save(self.model.state_dict(), "best_metric_model_swinunetr.pth")
            
            # Log best metrics
            self.log("best_metric", self.best_metric, prog_bar=False)
            self.log("best_metric_epoch", self.best_metric_epoch, prog_bar=False)
            
            print(f"\n🎯 New best model at epoch {self.best_metric_epoch}!")
            print(f"   Mean Dice: {self.best_metric:.4f}")
            print(f"   TC: {current_metrics['tc']:.4f}, WT: {current_metrics['wt']:.4f}, ET: {current_metrics['et']:.4f}")

    def on_train_epoch_end(self):
        # FIXED: Compute and log epoch averages for better tracking
        if self.train_metrics['loss']:
            avg_train_loss = sum(self.train_metrics['loss']) / len(self.train_metrics['loss'])
            avg_train_dice = sum(self.train_metrics['dice_mean']) / len(self.train_metrics['dice_mean'])
            
            self.log("epoch_train_loss", avg_train_loss, prog_bar=False)
            self.log("epoch_train_dice", avg_train_dice, prog_bar=False)
            
            # Clear epoch metrics
            for key in self.train_metrics:
                self.train_metrics[key].clear()

    def on_train_end(self):
        """Print final training summary"""
        print(f"\n🎉 Training completed!")
        print(f"   Best Mean Dice: {self.best_metric:.4f} at epoch {self.best_metric_epoch}")
        
        if self.val_metrics['dice_tc']:
            final_metrics = {
                'tc': self.val_metrics['dice_tc'][-1],
                'wt': self.val_metrics['dice_wt'][-1], 
                'et': self.val_metrics['dice_et'][-1]
            }
            print(f"   Final per-class Dice scores:")
            print(f"     - Tumor Core (TC): {final_metrics['tc']:.4f}")
            print(f"     - Whole Tumor (WT): {final_metrics['wt']:.4f}")
            print(f"     - Enhancing Tumor (ET): {final_metrics['et']:.4f}")

    def configure_optimizers(self):
        """Configure optimizer with improved learning rate scheduling"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # FIXED: Better learning rate scheduling with warmup and cosine annealing
        total_steps = len(self.train_loader) * self.hparams.max_epochs
        warmup_steps = len(self.train_loader) * self.hparams.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
            'name': 'learning_rate'
        }
        
        return [optimizer], [scheduler]