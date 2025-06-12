# architecture.py
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
        x = x[:, :D, :H, :W, :].contiguous()
        
    return x


class DepthwiseConv3D(nn.Module):
    """Efficient depthwise separable 3D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class EfficientWindowAttention3D(nn.Module):
    """Memory-efficient 3D Window Attention with linear complexity"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Efficient QKV projection with dimension reduction
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias if available
        if hasattr(self, 'relative_position_bias_table'):
            # Reshape relative position bias to match attention scores
            relative_position_bias = self.relative_position_bias_table.view(
                self.num_heads, -1
            ).permute(1, 0).contiguous()
            relative_position_bias = relative_position_bias.view(
                self.window_size * self.window_size * self.window_size,
                self.window_size * self.window_size * self.window_size,
                self.num_heads
            ).permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientMLP(nn.Module):
    """Efficient MLP with depthwise convolutions"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm3d(hidden_features)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        
        # Apply depthwise conv in spatial domain
        x_spatial = x.transpose(1, 2).view(B, C, D, H, W)
        x_spatial = self.dwconv(x_spatial)
        x_spatial = self.bn(x_spatial)
        x = x_spatial.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block with better efficiency"""
    def __init__(self, dim, num_heads, window_size=4, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientWindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EfficientMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x, D, H, W):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Store original dimensions
        original_D, original_H, original_W = D, H, W

        # Window partition
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

        # W-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, original_D, original_H, original_W)
        
        x = x.reshape(B, original_D * original_H * original_W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x), original_D, original_H, original_W)

        return x


class PatchEmbedding3D(nn.Module):
    """Patch embedding with better feature extraction"""
    def __init__(self, patch_size=4, in_chans=4, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Use depthwise separable conv for efficiency
        self.proj = nn.Sequential(
            DepthwiseConv3D(in_chans, embed_dim//2, kernel_size=7, stride=2, padding=3),
            DepthwiseConv3D(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, D//4, H//4, W//4
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, D*H*W, C
        x = self.norm(x)
        return x, (D, H, W)


class PatchMerging3D(nn.Module):
    """Patch merging with better downsampling"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Use depthwise separable conv for merging
        self.conv = DepthwiseConv3D(dim, 2 * dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x, D, H, W):
        B, L, C = x.shape
        
        # Reshape to spatial dimensions
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        # Apply convolution
        x = self.conv(x)  # B, 2*C, D//2, H//2, W//2
        
        B, C_new, D_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, D*H*W, 2*C
        x = self.norm(x)

        return x, (D_new, H_new, W_new)


class SwinEncoder3D(nn.Module):
    """Swin Transformer Encoder with better efficiency"""
    def __init__(
        self,
        patch_size=4,
        in_chans=4,
        embed_dim=64,  # Increased base dimension
        depths=[2, 2, 6, 2],  # Standard depths
        num_heads=[2, 4, 8, 16],  # More heads for better representation
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
    """SegFormer-style decoder with skip connections"""
    def __init__(
        self,
        feature_dims=[64, 128, 256, 512],
        decoder_embed_dim=128,  # Increased from 64
        num_classes=3,
        dropout=0.1
    ):
        super().__init__()
        
        # Feature projections
        self.linear_c4 = nn.Sequential(
            nn.Conv3d(feature_dims[3], decoder_embed_dim, 1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv3d(feature_dims[2], decoder_embed_dim, 1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv3d(feature_dims[1], decoder_embed_dim, 1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c1 = nn.Sequential(
            nn.Conv3d(feature_dims[0], decoder_embed_dim, 1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion with attention
        self.fuse_attention = nn.Sequential(
            nn.Conv3d(4 * decoder_embed_dim, decoder_embed_dim, 3, padding=1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_embed_dim, 4, 1),
            nn.Sigmoid()
        )
        
        # Enhanced fusion module
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(4 * decoder_embed_dim, decoder_embed_dim, 3, padding=1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_embed_dim, decoder_embed_dim, 3, padding=1),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
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
        
        # Concatenate features
        concat_features = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        
        # Attention-based fusion
        attention_weights = self.fuse_attention(concat_features)
        weighted_features = torch.cat([
            _c4 * attention_weights[:, 0:1], 
            _c3 * attention_weights[:, 1:2],
            _c2 * attention_weights[:, 2:3], 
            _c1 * attention_weights[:, 3:4]
        ], dim=1)
        
        # Fuse features
        fused = self.linear_fuse(weighted_features)
        fused = self.dropout(fused)
        
        # Final classification
        output = self.classifier(fused)
        
        # Upsample to original resolution
        output = torch.nn.functional.interpolate(
            output, scale_factor=4, mode="trilinear", align_corners=False
        )
        
        return output


class LightweightSwinUNETR(nn.Module):
    """Lightweight SwinUNETR with better performance"""
    def __init__(
        self,
        patch_size=4,
        in_channels=4,
        out_channels=3,
        embed_dim=64,  # Increased base dimension
        depths=[2, 2, 6, 2],  # Standard depths
        num_heads=[2, 4, 8, 16],  # More heads
        window_size=4,
        mlp_ratio=2.,
        decoder_embed_dim=128,  # Increased decoder dimension
        dropout=0.1
    ):
        super().__init__()
        
        # Encoder
        self.encoder = SwinEncoder3D(
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
        
        # Decoder
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
        learning_rate=1e-3,  # Reduced learning rate
        img_size=128,
        feature_size=48,
        embed_dim=64,  # Increased
        depths=[2, 2, 6, 2],  # Standard depths
        num_heads=[2, 4, 8, 16],  # More heads
        window_size=4,
        mlp_ratio=2.0,  # Reduced MLP ratio
        decoder_embed_dim=128,  # Increased
        patch_size=4,
        weight_decay=1e-4,
        warmup_epochs=10,  # More warmup
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
        
        # Combined loss function for better training
        self.dice_loss = DiceLoss(
            smooth_nr=0, 
            smooth_dr=1e-5, 
            squared_pred=True, 
            to_onehot_y=False, 
            sigmoid=True
        )
        self.ce_loss = nn.CrossEntropyLoss()
        
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
        
        # Combined loss
        dice_loss = self.dice_loss(outputs, labels)
        
        # Convert to one-hot for CE loss
        labels_onehot = torch.zeros_like(outputs)
        for i in range(3):
            labels_onehot[:, i] = (labels[:, 0] == i).float()
        
        ce_loss = self.ce_loss(outputs, labels_onehot.argmax(dim=1))
        loss = 0.7 * dice_loss + 0.3 * ce_loss
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice_loss", dice_loss, prog_bar=True)
        self.log("train_ce_loss", ce_loss, prog_bar=True)

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
        
        val_loss = self.dice_loss(val_outputs, val_labels)
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
            torch.save(self.model.state_dict(), "best_metric_model_swinunetr.pth")
            self.log("best_metric", self.best_metric)
    
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def on_train_end(self):
        print(f"Train completed, best_metric: {self.best_metric:.4f} at epoch: {self.best_metric_epoch}, "
              f"tc: {self.metric_values_tc[-1]:.4f}, "
              f"wt: {self.metric_values_wt[-1]:.4f}, "
              f"et: {self.metric_values_et[-1]:.4f}.")

    def configure_optimizers(self):
        # Use different learning rates for different components
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': self.hparams.learning_rate},
            {'params': decoder_params, 'lr': self.hparams.learning_rate * 2}  # Higher LR for decoder
        ], weight_decay=self.hparams.weight_decay)
        
        # Cosine annealing with warmup
        total_steps = len(self.train_loader) * self.hparams.max_epochs
        warmup_steps = len(self.train_loader) * self.hparams.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]


def test_model():
    """Test the model with architecture"""
    print("🧪 Testing model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(1, 4, 128, 128, 128).to(device)
    
    # Create model
    model = LightweightSwinUNETR(
        embed_dim=64,
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        decoder_embed_dim=128
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


# Comparison function
def compare_models():
    """Compare original vs model"""
    print("🔬 Comparing model architectures...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Original model (from your code)
    from architecture import LightweightSwinUNETR as OriginalModel
    original = OriginalModel(
        embed_dim=32,
        depths=[1, 1, 1, 1],
        decoder_embed_dim=64
    ).to(device)
    
    # Model
    model = LightweightSwinUNETR(
        embed_dim=64,
        depths=[2, 2, 6, 2],
        decoder_embed_dim=128
    ).to(device)
    
    original_params = original.count_parameters()
    model_params = model.count_parameters()
    
    print(f"📊 Model Comparison:")
    print(f"   Original Model: {original_params:,} parameters ({original_params/1e6:.2f}M)")
    print(f"   Model: {model_params:,} parameters ({model_params/1e6:.2f}M)")
    print(f"   Parameter increase: {((model_params/original_params)-1)*100:.1f}%")
    
    # Test inference time
    test_input = torch.randn(1, 4, 96, 96, 96).to(device)
    
    # Original model timing
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        _ = original(test_input)
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    
    # Model timing
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        _ = model(test_input)
    torch.cuda.synchronize()
    model_time = time.time() - start_time
    
    print(f"⏱️  Inference Time Comparison (96³ volume):")
    print(f"   Original Model: {original_time*1000:.1f}ms")
    print(f"   Model: {model_time*1000:.1f}ms")
    print(f"   Time increase: {((model_time/original_time)-1)*100:.1f}%")


if __name__ == "__main__":
    # Test the model
    success = test_model()
    if success:
        print("🎉 Model is ready for training!")
        compare_models()
    else:
        print("💥 Model needs debugging.")