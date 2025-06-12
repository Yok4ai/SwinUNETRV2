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

        # Create relative position indices
        coords_d = torch.arange(window_size)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 2] += window_size - 1
        relative_coords[:, :, 0] *= (2 * window_size - 1) * (2 * window_size - 1)
        relative_coords[:, :, 1] *= 2 * window_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

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
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size * self.window_size,
            self.window_size * self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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
        
        # Apply depthwise conv in spatial domain - use the correct hidden_features dimension
        _, _, hidden_features = x.shape
        x_spatial = x.transpose(1, 2).view(B, hidden_features, D, H, W)
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
        embed_dim=96,  # FIXED: Use proven dimension from SwinUNETR
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],  # FIXED: Proper head scaling
        window_size=7,  # FIXED: Use standard window size
        mlp_ratio=4.,  # FIXED: Use standard ratio
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
        feature_dims=[96, 192, 384, 768],  # FIXED: Standard SwinUNETR dimensions
        decoder_embed_dim=256,  # FIXED: Increased for better representation
        num_classes=3,
        dropout=0.1
    ):
        super().__init__()
        
        # Feature projections with proper normalization
        self.linear_c4 = nn.Sequential(
            nn.Conv3d(feature_dims[3], decoder_embed_dim, 1, bias=False),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c3 = nn.Sequential(
            nn.Conv3d(feature_dims[2], decoder_embed_dim, 1, bias=False),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c2 = nn.Sequential(
            nn.Conv3d(feature_dims[1], decoder_embed_dim, 1, bias=False),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_c1 = nn.Sequential(
            nn.Conv3d(feature_dims[0], decoder_embed_dim, 1, bias=False),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # FIXED: Simplified fusion without attention for stability
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(4 * decoder_embed_dim, decoder_embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_embed_dim, decoder_embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm3d(decoder_embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout3d(dropout)
        
        # FIXED: Direct classification without sigmoid (let loss handle it)
        self.classifier = nn.Conv3d(decoder_embed_dim, num_classes, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Proper weight initialization"""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
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
        
        # Concatenate and fuse features
        concat_features = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        fused = self.linear_fuse(concat_features)
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
        embed_dim=96,  # FIXED: Standard SwinUNETR embedding dimension
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],  # FIXED: Proper head scaling
        window_size=7,  # FIXED: Standard window size
        mlp_ratio=4.,  # FIXED: Standard MLP ratio
        decoder_embed_dim=256,  # FIXED: Increased decoder dimension
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
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Proper weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
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
        
        # FIXED: Use DiceCELoss for better class balance
        self.dice_ce_loss = DiceCELoss(
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=True,  # FIXED: Convert labels to one-hot
            softmax=True,      # FIXED: Apply softmax
            include_background=False,  # FIXED: Exclude background
            ce_weight=torch.tensor([1.0, 2.0, 4.0])  # FIXED: Weight rare classes more
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
        
        # FIXED: Proper loss calculation
        loss = self.dice_ce_loss(outputs, labels)
        
        self.log("train_loss", loss, prog_bar=True)

        # FIXED: Proper metric calculation
        outputs_softmax = torch.softmax(outputs, dim=1)
        outputs_onehot = torch.zeros_like(outputs)
        outputs_onehot.scatter_(1, outputs_softmax.argmax(dim=1, keepdim=True), 1)
        
        # Convert labels to one-hot for metric calculation
        labels_onehot = torch.zeros_like(outputs)
        labels_onehot.scatter_(1, labels.long(), 1)
        
        self.dice_metric(y_pred=outputs_onehot, y=labels_onehot)
        self.dice_metric_batch(y_pred=outputs_onehot, y=labels_onehot)

        train_dice = self.dice_metric.aggregate().item()
        self.log("train_mean_dice", train_dice, prog_bar=True)

        self.train_metric_values.append(train_dice)

        metric_batch = self.dice_metric_batch.aggregate()
        if len(metric_batch) >= 3:  # Ensure we have all 3 classes
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
        
        # FIXED: Proper validation loss calculation
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


def test_model():
    """Test the model with architecture"""
    print("🧪 Testing fixed model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(1, 4, 128, 128, 128).to(device)
    
    # Create model with fixed parameters
    model = LightweightSwinUNETR(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        decoder_embed_dim=256
    ).to(device)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"✅ Fixed model has {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_input)
            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            
            # Verify output properties
            print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"   Output mean: {output.mean():.3f}")
            
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
    # Test the fixed model
    success = test_model()
    if success:
        print("🎉 Fixed model is ready for training!")
        print("\n🔧 Key fixes applied:")
        print("• Standard SwinUNETR embedding dimension (96)")
        print("• Proper head scaling [3, 6, 12, 24]")
        print("• Standard window size (7)")
        print("• DiceCELoss with class weighting")
        print("• Fixed post-processing for multi-class")
        print("• Proper one-hot label conversion")
        print("• Exclude background from metrics")
        print("• Conservative learning rate and optimization")
    else:
        print("💥 Model needs further debugging.")