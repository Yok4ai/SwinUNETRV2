import os
import time
import torch
import torch.nn as nn
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
from typing import Tuple, Union
import math
from einops import rearrange


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding if needed"""
    B, D, H, W, C = x.shape
    
    # Calculate padding needed
    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    # Apply padding if needed
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        
    B, D, H, W, C = x.shape  # Update dimensions after padding
    
    x = x.view(B, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    """Reverse window partition and remove padding"""
    # Calculate padded dimensions
    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    D_pad = D + pad_d
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    B = int(windows.shape[0] / (D_pad * H_pad * W_pad / window_size / window_size / window_size))
    x = windows.view(B, D_pad // window_size, H_pad // window_size, W_pad // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D_pad, H_pad, W_pad, -1)
    
    # Remove padding if it was added
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = x[:, :D, :H, :W, :].contiguous()  # Add contiguous() here
        
    return x


class WindowAttention3D(nn.Module):
    """Lightweight 3D Window Attention"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Lightweight QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Lightweight Swin Transformer Block for 3D"""
    def __init__(self, dim, num_heads, window_size=4, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Lightweight MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, D, H, W):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Store original dimensions for padding removal
        original_D, original_H, original_W = D, H, W

        # Window partition (with automatic padding)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

        # W-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows (with automatic padding removal)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, original_D, original_H, original_W)
        
        # FIX: Use reshape instead of view for non-contiguous tensors
        x = x.reshape(B, original_D * original_H * original_W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbedding3D(nn.Module):
    """Lightweight 3D Patch Embedding"""
    def __init__(self, patch_size=4, in_chans=4, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, D//4, H//4, W//4
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, D*H*W, C
        x = self.norm(x)
        return x, (D, H, W)


class PatchMerging3D(nn.Module):
    """Lightweight patch merging for downsampling with robust padding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)  # 2x3D = 8 neighbors
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x, D, H, W):
        B, L, C = x.shape
        
        x = x.view(B, D, H, W, C)
        
        # Always pad to ensure even dimensions
        pad_d = D % 2
        pad_h = H % 2  
        pad_w = W % 2
        
        if pad_d or pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = D + pad_d, H + pad_h, W + pad_w

        # Merge 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
        x4 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        # FIX: Use reshape instead of view
        x = x.reshape(B, -1, 8 * C)  # B D/2*H/2*W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, (D // 2, H // 2, W // 2)


class LightweightSwinEncoder3D(nn.Module):
    """Ultra-lightweight Swin Transformer Encoder for 3D"""
    def __init__(
        self,
        patch_size=4,
        in_chans=4,
        embed_dim=32,  # Much smaller base dimension
        depths=[1, 1, 1, 1],  # Fewer layers
        num_heads=[1, 2, 4, 8],  # Fewer heads
        window_size=4,
        mlp_ratio=2.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            # Add patch merging except for the last layer
            if i_layer < len(depths) - 1:
                self.layers.append(PatchMerging3D(layer_dim))
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # Patch embedding
        x, (D, H, W) = self.patch_embed(x)
        
        features = []
        dims = [(D, H, W)]
        
        layer_idx = 0
        for i in range(self.num_layers):
            # Transformer blocks
            for blk in self.layers[layer_idx]:
                x = blk(x, dims[-1][0], dims[-1][1], dims[-1][2])
            
            # Store feature for decoder
            B, L, C = x.shape
            D, H, W = dims[-1]
            # FIX: Use reshape and ensure contiguous
            feature = x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
            features.append(feature)
            
            layer_idx += 1
            
            # Patch merging (except last layer)
            if i < self.num_layers - 1:
                x, (D, H, W) = self.layers[layer_idx](x, dims[-1][0], dims[-1][1], dims[-1][2])
                dims.append((D, H, W))
                layer_idx += 1
        
        return features


class SegFormerDecoder3D(nn.Module):
    """Ultra-lightweight SegFormer-style decoder"""
    def __init__(
        self,
        feature_dims=[32, 64, 128, 256],  # Much smaller dimensions
        decoder_embed_dim=64,  # Reduced from 256
        num_classes=3,
        dropout=0.1
    ):
        super().__init__()
        
        # Lightweight linear projections
        self.linear_c4 = nn.Conv3d(feature_dims[3], decoder_embed_dim, 1)
        self.linear_c3 = nn.Conv3d(feature_dims[2], decoder_embed_dim, 1) 
        self.linear_c2 = nn.Conv3d(feature_dims[1], decoder_embed_dim, 1)
        self.linear_c1 = nn.Conv3d(feature_dims[0], decoder_embed_dim, 1)
        
        # Lightweight fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(4 * decoder_embed_dim, decoder_embed_dim, 1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout3d(dropout)
        self.classifier = nn.Conv3d(decoder_embed_dim, num_classes, 1)
        
    def forward(self, features):
        c1, c2, c3, c4 = features
        
        # Linear projections
        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)  
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)
        
        # Upsample to c1 size
        target_size = c1.size()[2:]
        
        _c4 = torch.nn.functional.interpolate(_c4, size=target_size, mode="trilinear", align_corners=False)
        _c3 = torch.nn.functional.interpolate(_c3, size=target_size, mode="trilinear", align_corners=False)
        _c2 = torch.nn.functional.interpolate(_c2, size=target_size, mode="trilinear", align_corners=False)
        
        # Fuse features
        fused = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        fused = self.dropout(fused)
        
        # Final classification
        output = self.classifier(fused)
        
        # Upsample to original resolution (4x upsampling to match patch_size=4)
        output = torch.nn.functional.interpolate(
            output, scale_factor=4, mode="trilinear", align_corners=False
        )
        
        return output


class LightweightSwinUNETR(nn.Module):
    """Ultra-lightweight SwinUNETR with ~4M parameters"""
    def __init__(
        self,
        patch_size=4,
        in_channels=4,
        out_channels=3,
        embed_dim=32,  # Very small base dimension
        depths=[1, 1, 1, 1],  # Minimal depth
        num_heads=[1, 2, 4, 8],
        window_size=4,
        mlp_ratio=2.,
        decoder_embed_dim=64,  # Small decoder
        dropout=0.1
    ):
        super().__init__()
        
        # Lightweight encoder
        self.encoder = LightweightSwinEncoder3D(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )
        
        # Calculate feature dimensions
        feature_dims = [int(embed_dim * 2 ** i) for i in range(len(depths))]
        
        # Lightweight decoder
        self.decoder = SegFormerDecoder3D(
            feature_dims=feature_dims,
            decoder_embed_dim=decoder_embed_dim,
            num_classes=out_channels,
            dropout=dropout
        )
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BrainTumorSegmentation(pl.LightningModule):
    def __init__(
        self, 
        train_loader, 
        val_loader, 
        max_epochs=50, 
        val_interval=1, 
        learning_rate=2e-3,
        img_size=128,
        feature_size=48,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        decoder_embed_dim=256,
        patch_size=4,
        weight_decay=1e-4,
        warmup_epochs=5,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        roi_size=(128, 128, 128),
        sw_batch_size=2,
        overlap=0.25
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Ultra-lightweight model
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
        loss = self.loss_function(outputs, labels)
        
        self.log("train_loss", loss, prog_bar=True)

        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
        
        self.dice_metric(y_pred=outputs, y=labels)
        self.dice_metric_batch(y_pred=outputs, y=labels)

        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True)

        self.train_metric_values.append(train_dice)

        metric_batch = self.dice_metric_batch.aggregate()
        self.train_metric_values_tc.append(metric_batch[0].item())
        self.train_metric_values_wt.append(metric_batch[1].item())
        self.train_metric_values_et.append(metric_batch[2].item())

        self.log("train_tc", metric_batch[0].item(), prog_bar=True)
        self.log("train_wt", metric_batch[1].item(), prog_bar=True)
        self.log("train_et", metric_batch[2].item(), prog_bar=True)

        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        return loss

    def on_train_epoch_end(self):
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
        
        val_loss = self.loss_function(val_outputs, val_labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]    
        
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        self.dice_metric_batch(y_pred=val_outputs, y=val_labels)
    
        val_dice = self.dice_metric.aggregate().item()
        self.log("val_mean_dice", val_dice, prog_bar=True)
    
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_dice = self.dice_metric.aggregate().item()
        self.metric_values.append(val_dice)

        val_loss = self.trainer.logged_metrics["val_loss"].item()
        self.epoch_loss_values.append(val_loss)

        avg_val_loss = sum(self.epoch_loss_values) / len(self.epoch_loss_values)
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.avg_val_loss_values.append(avg_val_loss)

        metric_batch = self.dice_metric_batch.aggregate()
        self.metric_values_tc.append(metric_batch[0].item())
        self.metric_values_wt.append(metric_batch[1].item())
        self.metric_values_et.append(metric_batch[2].item())

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mean_dice", val_dice, prog_bar=True)
        self.log("val_tc", metric_batch[0].item(), prog_bar=True)
        self.log("val_wt", metric_batch[1].item(), prog_bar=True)
        self.log("val_et", metric_batch[2].item(), prog_bar=True)
    
        if val_dice > self.best_metric:
            self.best_metric = val_dice
            self.best_metric_epoch = self.current_epoch
            torch.save(self.model.state_dict(), "best_metric_model_lightweight_swinunetr.pth")
            self.log("best_metric", self.best_metric)
    
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def on_train_end(self):
        print(f"Train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}, "
              f"tc: {self.metric_values_tc[-1]:.4f}, "
              f"wt: {self.metric_values_wt[-1]:.4f}, "
              f"et: {self.metric_values_et[-1]:.4f}.")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Add warmup scheduler
        total_steps = len(self.train_loader) * self.hparams.max_epochs
        warmup_steps = len(self.train_loader) * self.hparams.warmup_epochs
        
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]


# Memory-efficient model test function
def test_model_memory():
    """Test the model with a small input to verify it works"""
    print("🧪 Testing model memory and functionality...")
    
    # Create a small test input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(1, 4, 96, 96, 96).to(device)
    
    # Create model
    model = LightweightSwinUNETR(
        embed_dim=32,
        depths=[1, 1, 1, 1],
        decoder_embed_dim=64
    ).to(device)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"✅ Model has {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_input)
            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                print(f"   GPU memory used: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
    
    return True


if __name__ == "__main__":
    # Test the model
    success = test_model_memory()
    if success:
        print("🎉 Model is ready for training!")
    else:
        print("💥 Model needs debugging before training.")