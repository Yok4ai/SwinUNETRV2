#pipeline.py
import torch
import pytorch_lightning as pl
import math
from monai.losses import DiceCELoss
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
        learning_rate=5e-4,  # FIXED: More conservative learning rate
        img_size=128,
        feature_size=48,
        embed_dim=96,  # FIXED: Standard dimension
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],  # FIXED: Proper head scaling
        window_size=7,  # FIXED: Standard window size
        mlp_ratio=4.0,  # FIXED: Standard ratio
        decoder_embed_dim=256,  # FIXED: Larger decoder
        patch_size=4,
        weight_decay=1e-5,  # FIXED: Reduced weight decay
        warmup_epochs=5,  # FIXED: Reduced warmup
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
        
        # FIXED: Handle class weights separately
        self.class_weights = torch.tensor([1.0, 2.0, 4.0]).to(self.device)
        
        # FIXED: Use DiceCELoss for better class balance
        self.dice_ce_loss = DiceCELoss(
            include_background=False,  # FIXED: Exclude background
            to_onehot_y=True,  # FIXED: Convert labels to one-hot
            softmax=True,      # FIXED: Apply softmax
            weight=self.class_weights,  # FIXED: Use class weights for both losses
            lambda_dice=1.0,   # FIXED: Weight for dice component
            lambda_ce=1.0,     # FIXED: Weight for CE component
            smooth_nr=1e-5,    # FIXED: Small constant for numerator
            smooth_dr=1e-5     # FIXED: Small constant for denominator
        )
        
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")  # FIXED: Exclude background
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")  # FIXED: Exclude background
        
        # FIXED: Proper post-processing for multi-class
        self.post_trans = Compose([
            Activations(softmax=True),
            AsDiscrete(argmax=True, to_onehot=3)
        ])
        
        self.best_metric = -1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]

        outputs = self(inputs)
        
        # FIXED: Ensure labels have correct shape for DiceCELoss
        if labels.shape[1] != 1:
            labels = labels.argmax(dim=1, keepdim=True)
        
        # FIXED: Proper loss calculation with class weights
        loss = self.dice_ce_loss(outputs, labels)
        
        self.log("train_loss", loss, prog_bar=True)

        # FIXED: Proper metric calculation
        outputs_softmax = torch.softmax(outputs, dim=1)
        
        # Convert labels to one-hot for metric calculation
        labels_onehot = torch.zeros_like(outputs)
        labels_onehot.scatter_(1, labels.long(), 1)
        
        # Use softmax outputs directly for metric calculation
        self.dice_metric(y_pred=outputs_softmax, y=labels_onehot)
        self.dice_metric_batch(y_pred=outputs_softmax, y=labels_onehot)

        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True)

        self.train_metric_values.append(train_dice)

        metric_batch = self.dice_metric_batch.aggregate()
        if len(metric_batch) >= 3:  # Ensure we have all 3 classes
            self.train_metric_values_tc.append(metric_batch[0].item())
            self.train_metric_values_wt.append(metric_batch[1].item())
            self.train_metric_values_et.append(metric_batch[2].item())

            # Log all dice scores in progress bar
            self.log("train_tc", metric_batch[0].item(), prog_bar=True)
            self.log("train_wt", metric_batch[1].item(), prog_bar=True)
            self.log("train_et", metric_batch[2].item(), prog_bar=True)

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return loss

    def on_train_epoch_end(self):
        if hasattr(self.trainer, 'logged_metrics') and "train_loss" in self.trainer.logged_metrics:
            train_loss = self.trainer.logged_metrics["train_loss"].item()
            self.train_loss_values.append(train_loss)
            
            avg_train_loss = sum(self.train_loss_values) / len(self.train_loss_values)
            self.log("avg_train_loss", avg_train_loss, prog_bar=True)
            self.avg_train_loss_values.append(avg_train_loss)

    def validation_step(self, batch, batch_idx):
        val_inputs, val_labels = batch["image"], batch["label"]
        
        val_outputs = sliding_window_inference(
            val_inputs, 
            roi_size=self.roi_size, 
            sw_batch_size=self.sw_batch_size, 
            predictor=self.model,
            overlap=self.overlap
        )
        
        # FIXED: Ensure labels have correct shape for DiceCELoss
        if val_labels.shape[1] != 1:
            val_labels = val_labels.argmax(dim=1, keepdim=True)
        
        # FIXED: Proper validation loss calculation with class weights
        val_loss = self.dice_ce_loss(val_outputs, val_labels)
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        # FIXED: Proper validation metric calculation
        val_outputs_softmax = torch.softmax(val_outputs, dim=1)
        val_outputs_onehot = torch.zeros_like(val_outputs)
        val_outputs_onehot.scatter_(1, val_outputs_softmax.argmax(dim=1, keepdim=True), 1)
        
        # Convert labels to one-hot for metric calculation
        val_labels_onehot = torch.zeros_like(val_outputs)
        val_labels_onehot.scatter_(1, val_labels.long(), 1)
        
        self.dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)
        self.dice_metric_batch(y_pred=val_outputs_onehot, y=val_labels_onehot)
    
        val_dice = self.dice_metric.aggregate().item()
        self.log("val_mean_dice", val_dice, prog_bar=True)

        # Log all validation dice scores in progress bar
        metric_batch = self.dice_metric_batch.aggregate()
        if len(metric_batch) >= 3:  # Ensure we have all 3 classes
            self.log("val_tc", metric_batch[0].item(), prog_bar=True)
            self.log("val_wt", metric_batch[1].item(), prog_bar=True)
            self.log("val_et", metric_batch[2].item(), prog_bar=True)
    
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_dice = self.dice_metric.aggregate().item()
        self.metric_values.append(val_dice)

        if hasattr(self.trainer, 'logged_metrics') and "val_loss" in self.trainer.logged_metrics:
            val_loss = self.trainer.logged_metrics["val_loss"].item()
            self.epoch_loss_values.append(val_loss)

            avg_val_loss = sum(self.epoch_loss_values) / len(self.epoch_loss_values)
            self.log("avg_val_loss", avg_val_loss, prog_bar=True)
            self.avg_val_loss_values.append(avg_val_loss)

        metric_batch = self.dice_metric_batch.aggregate()
        if len(metric_batch) >= 3:  # Ensure we have all 3 classes
            self.metric_values_tc.append(metric_batch[0].item())
            self.metric_values_wt.append(metric_batch[1].item())
            self.metric_values_et.append(metric_batch[2].item())

            self.log("val_tc", metric_batch[0].item(), prog_bar=True)
            self.log("val_wt", metric_batch[1].item(), prog_bar=True)
            self.log("val_et", metric_batch[2].item(), prog_bar=True)
    
        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_metric_model_swinunetr.pth")
            self.log("best_metric", self.best_metric)
    
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def on_train_end(self):
        print(f"Train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}")
        if len(self.metric_values_tc) > 0:
            print(f"Final metrics - TC: {self.metric_values_tc[-1]:.4f}, "
                  f"WT: {self.metric_values_wt[-1]:.4f}, "
                  f"ET: {self.metric_values_et[-1]:.4f}")

    def configure_optimizers(self):
        # FIXED: More conservative optimization strategy
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # FIXED: Simpler scheduler with warmup
        total_steps = len(self.train_loader) * self.hparams.max_epochs
        warmup_steps = len(self.train_loader) * self.hparams.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps))))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]