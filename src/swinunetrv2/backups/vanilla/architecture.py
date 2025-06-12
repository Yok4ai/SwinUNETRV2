import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class BrainTumorModel(nn.Module):
    def __init__(
        self,
        train_loader=None,
        val_loader=None,
        max_epochs=30,
        learning_rate=1e-4,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        patch_size=(2, 2, 2),
        weight_decay=1e-5,
        warmup_epochs=5,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.5
    ):
        super().__init__()
        # SwinUNETR-V2 Configuration
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,  # Enable SwinUNETR-V2!
            spatial_dims=3,
            depths=depths,
            num_heads=num_heads,
            norm_name="instance",
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=0.0,
            downsample="mergingv2"  # Use improved merging for V2
        )
    
    def forward(self, x):
        return self.model(x)