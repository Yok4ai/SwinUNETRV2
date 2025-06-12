# architecture.py
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

class ParameterEfficientSwinUNETR(nn.Module):
    """Wrapper around MONAI SwinUNETR with parameter reduction techniques"""
    
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        feature_size=32,  # Reduced for parameter efficiency
        use_checkpoint=True,
        use_v2=True,
        spatial_dims=3,
        depths=(2, 2, 2, 2),  # Reduced depth for efficiency
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.1,
        attn_drop_rate=0.1,
        dropout_path_rate=0.1,
        downsample="mergingv2"
    ):
        super().__init__()
        
        # Use MONAI's proven SwinUNETR
        self.swinunetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            use_v2=use_v2,
            spatial_dims=spatial_dims,
            depths=depths,
            num_heads=num_heads,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            downsample=downsample
        )
        
        print(f"✅ MONAI SwinUNETR initialized with feature_size={feature_size}, depths={depths}")
        
    def forward(self, x):
        return self.swinunetr(x)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params
