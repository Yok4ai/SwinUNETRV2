import os
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.networks.nets import SwinUNETR
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
from typing import Tuple, Union


class SegFormerDecoder3D(nn.Module):
    """
    SegFormer-style decoder for 3D medical image segmentation.
    Uses linear projections and upsampling instead of complex 3D convolutions.
    """
    def __init__(
        self,
        feature_dims: list = [768, 384, 192, 96],  # SwinUNETR feature dimensions
        decoder_embed_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Linear projections for each feature level (step 1)
        self.linear_c4 = self._make_linear_projection(feature_dims[0], decoder_embed_dim)
        self.linear_c3 = self._make_linear_projection(feature_dims[1], decoder_embed_dim) 
        self.linear_c2 = self._make_linear_projection(feature_dims[2], decoder_embed_dim)
        self.linear_c1 = self._make_linear_projection(feature_dims[3], decoder_embed_dim)
        
        # Fusion layer (step 3)
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_embed_dim,
                out_channels=decoder_embed_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout3d(dropout)
        
        # Final classification head (step 4)
        self.classifier = nn.Conv3d(
            decoder_embed_dim, 
            num_classes, 
            kernel_size=1
        )
        
        # Upsampling to original resolution
        self.final_upsample = nn.Upsample(
            scale_factor=2.0,  # Adjust based on your input/output ratio
            mode="trilinear", 
            align_corners=False
        )
        
    def _make_linear_projection(self, input_dim: int, output_dim: int):
        """Create linear projection with layer norm"""
        return nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=1),
            nn.BatchNorm3d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: list):
        """
        Args:
            features: List of feature maps from encoder [c1, c2, c3, c4]
                     where c1 is highest resolution, c4 is lowest
        """
        c1, c2, c3, c4 = features
        
        # Step 1: Linear projection to fixed dimension
        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)  
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)
        
        # Step 2: Upsample all features to c1 size (highest resolution)
        target_size = c1.size()[2:]  # (D, H, W)
        
        _c4 = torch.nn.functional.interpolate(
            _c4, size=target_size, mode="trilinear", align_corners=False
        )
        _c3 = torch.nn.functional.interpolate(
            _c3, size=target_size, mode="trilinear", align_corners=False  
        )
        _c2 = torch.nn.functional.interpolate(
            _c2, size=target_size, mode="trilinear", align_corners=False
        )
        # _c1 is already at target size
        
        # Step 3: Concatenate and fuse
        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        fused = self.dropout(fused)
        
        # Step 4: Generate segmentation masks
        output = self.classifier(fused)
        output = self.final_upsample(output)
        
        return output


class SwinUNETRWithSegFormerDecoder(nn.Module):
    """
    SwinUNETR encoder with SegFormer-style decoder for efficient 3D segmentation
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        decoder_embed_dim: int = 256,
        decoder_dropout: float = 0.1
    ):
        super().__init__()
        
        # Use SwinUNETR as encoder (extract features only)
        self.encoder = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,  # We'll override this with our decoder
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            use_v2=True,
            spatial_dims=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            downsample="mergingv2"
        )
        
        # Calculate feature dimensions based on feature_size
        # SwinUNETR uses feature_size * (1, 2, 4, 8) for the four stages
        feature_dims = [
            feature_size,      # Stage 1
            feature_size * 2,  # Stage 2  
            feature_size * 4,  # Stage 3
            feature_size * 8   # Stage 4
        ]
        
        # SegFormer-style decoder
        self.decoder = SegFormerDecoder3D(
            feature_dims=feature_dims[::-1],  # Reverse order (c4, c3, c2, c1)
            decoder_embed_dim=decoder_embed_dim,
            num_classes=out_channels,
            dropout=decoder_dropout
        )
        
    def extract_features(self, x):
        """Extract multi-scale features from SwinUNETR encoder"""
        # We need to modify this to extract intermediate features
        # This is a simplified version - you may need to modify SwinUNETR source
        # to properly extract intermediate features
        
        # For now, we'll use the existing SwinUNETR but this is not optimal
        # Ideally, you'd modify the SwinUNETR to return intermediate features
        hidden_states_out = self.encoder.swinViT(x, normalize=True)
        
        # Extract features at different scales
        # Note: You might need to adjust these indices based on actual SwinUNETR implementation
        enc0 = hidden_states_out[0]  # Highest resolution
        enc1 = hidden_states_out[1] 
        enc2 = hidden_states_out[2]
        enc3 = hidden_states_out[3]  # Lowest resolution
        
        return [enc0, enc1, enc2, enc3]
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.extract_features(x)
        
        # Decode with SegFormer-style decoder
        output = self.decoder(features)
        
        return output


class BrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        max_epochs=100, 
        val_interval=1, 
        learning_rate=1e-4,
        feature_size=48,
        decoder_embed_dim=256,
        decoder_dropout=0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model with SegFormer decoder
        self.model = SwinUNETRWithSegFormerDecoder(
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            use_checkpoint=True,
            decoder_embed_dim=decoder_embed_dim,
            decoder_dropout=decoder_dropout
        )
        
        # Loss and metrics (same as before)
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

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]

        # Forward pass
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
        
        # Use sliding window inference for validation
        val_outputs = sliding_window_inference(
            val_inputs, 
            roi_size=(96, 96, 96), 
            sw_batch_size=1, 
            predictor=self.model,  # Use our custom model
            overlap=0.5
        )
        
        # Compute loss
        val_loss = self.loss_function(val_outputs, val_labels)
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
            torch.save(self.model.state_dict(), "best_metric_model_swinunetr_segformer.pth")
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
        # Slightly higher learning rate for the new architecture
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=1e-5
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]


# Example usage and configuration
# def create_model_with_segformer_decoder(train_loader, val_loader):
#     """
#     Factory function to create the model with optimized parameters
#     """
#     model = BrainTumorSegmentation(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         max_epochs=100,
#         learning_rate=2e-4,  # Slightly higher LR for the new architecture
#         feature_size=48,
#         decoder_embed_dim=256,  # SegFormer decoder embedding dimension
#         decoder_dropout=0.1
#     )
#     return model