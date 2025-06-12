import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class BrainTumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # SwinUNETR-V2 Configuration
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
            use_v2=True,  # Enable SwinUNETR-V2!
            spatial_dims=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            downsample="mergingv2"  # Use improved merging for V2
        )
    
    def forward(self, x):
        return self.model(x)