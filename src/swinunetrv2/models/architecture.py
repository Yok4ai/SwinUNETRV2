#architecture.py
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class BrainTumorModel(nn.Module):
    def __init__(self, args):
        """
        Brain Tumor Segmentation Model using SwinUNETR-V2
        
        Args:
            args: Argument namespace containing all configuration parameters
        """
        super().__init__()
        
        # Store args for access to all parameters
        self.args = args
        
        # SwinUNETR-V2 Configuration using parameters from args
        self.model = SwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            # use_v2=True,  # Enable SwinUNETR-V2!
            # spatial_dims=3,
            # norm_name="instance",
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            dropout_path_rate=0.0,
            # downsample="merging"
            # downsample="mergingv2"  # Use improved merging for V2
        )
    
    def forward(self, x):
        return self.model(x)


# Alternative: Direct SwinUNETR factory function for cleaner usage
def create_swinunetr_model(args):
    """
    Factory function to create SwinUNETR model with args
    
    Args:
        args: Argument namespace containing all configuration parameters
    
    Returns:
        SwinUNETR model configured with parameters from args
    """
    return SwinUNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        # use_v2=True,  # Enable SwinUNETR-V2!
        # spatial_dims=3,
        # norm_name="instance",
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=0.0,
        # downsample="mergingv2"  # Use improved merging for V2
    )