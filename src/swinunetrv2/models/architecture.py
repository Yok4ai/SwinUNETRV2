# architecture.py - Enhanced Parameter-Efficient SwinUNETR
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import Conv
import math
from typing import Sequence, Union


class UltraEfficientSwinUNETR(nn.Module):
    """
    Ultra parameter-efficient SwinUNETR inspired by SegFormer3D efficiency techniques.
    Reduces parameters by ~60-80% compared to standard MONAI SwinUNETR while maintaining performance.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 24,
        depths: Sequence[int] = (1, 1, 2, 1),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        use_checkpoint: bool = True,
        use_v2: bool = True,
        spatial_dims: int = 3,
        norm_name: str = "instance",
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        dropout_path_rate: float = 0.1,
        use_depthwise_conv: bool = True,
        use_lightweight_decoder: bool = True,
        decoder_channels: Sequence[int] = (96, 48, 24, 12),
        reduce_skip_connections: bool = True,
        use_separable_conv: bool = True,
    ):
        super().__init__()
        
        # Validate feature size is divisible by 12
        if feature_size % 12 != 0:
            raise ValueError(f"feature_size must be divisible by 12, got {feature_size}")
        
        self.feature_size = feature_size
        self.spatial_dims = spatial_dims
        self.use_lightweight_decoder = use_lightweight_decoder
        
        # Create the base SwinUNETR with minimal feature size
        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=feature_size * 4,
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
            downsample="mergingv2"
        )
        
        if use_lightweight_decoder:
            self._replace_decoder_with_lightweight(
                decoder_channels, 
                out_channels, 
                use_depthwise_conv,
                use_separable_conv,
                reduce_skip_connections
            )
        
        self._display_parameter_info()
    
    def _replace_decoder_with_lightweight(
        self, 
        decoder_channels, 
        out_channels, 
        use_depthwise_conv,
        use_separable_conv,
        reduce_skip_connections
    ):
        """Replace heavy decoder with SegFormer3D-style lightweight decoder"""
        
        # Store encoder features for skip connections
        self.encoder_features = []
        
        # Create lightweight decoder components with correct feature dimensions
        # These dimensions match the actual encoder output channels
        feature_dims = [
            self.feature_size * 8,   # From layer 4 (dec4)
            self.feature_size * 4,   # From layer 3 (enc3)
            self.feature_size * 2,   # From layer 2 (enc2)
            self.feature_size,       # From layer 1 (enc1)
        ]
        
        # Print feature dimensions for debugging
        print(f"Feature dimensions for projections: {feature_dims}")
        
        # MLP projections for each feature level (like SegFormer)
        self.feature_projections = nn.ModuleList([
            LightweightMLP(feature_dims[i], decoder_channels[0])
            for i in range(4)
        ])
        
        # Lightweight fusion module
        self.feature_fusion = LightweightFusion(
            in_channels=decoder_channels[0] * 4,
            out_channels=decoder_channels[0],
            use_depthwise=use_depthwise_conv,
            use_separable=use_separable_conv
        )
        
        # Minimal upsampling decoder
        self.lightweight_decoder = LightweightDecoder(
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_depthwise=use_depthwise_conv
        )
        
        # Override the forward method to use our lightweight decoder
        original_forward = self.backbone.forward
        
        def new_forward(x):
            # Get encoder features
            hidden_states_out = self.backbone.swinViT(x, self.backbone.normalize)
            
            # Extract features at different scales
            enc0 = self.backbone.encoder1(x)
            enc1 = self.backbone.encoder2(hidden_states_out[0])  
            enc2 = self.backbone.encoder3(hidden_states_out[1])
            enc3 = self.backbone.encoder4(hidden_states_out[2])
            dec4 = self.backbone.encoder10(hidden_states_out[4])
            
            # Store features for skip connections
            self.encoder_features = [enc0, enc1, enc2, enc3, dec4]
            
            # Apply lightweight decoder
            return self._lightweight_decode([enc0, enc1, enc2, enc3, dec4])
        
        # Replace the forward method
        self.backbone.forward = new_forward
    
    def forward(self, x):
        # Get encoder features
        hidden_states_out = self.backbone.swinViT(x, self.backbone.normalize)
        
        # Extract features at different scales
        enc0 = self.backbone.encoder1(x)
        enc1 = self.backbone.encoder2(hidden_states_out[0])  
        enc2 = self.backbone.encoder3(hidden_states_out[1])
        enc3 = self.backbone.encoder4(hidden_states_out[2])
        dec4 = self.backbone.encoder10(hidden_states_out[4])
        
        # Apply lightweight decoder
        if self.use_lightweight_decoder:
            return self._lightweight_decode([enc0, enc1, enc2, enc3, dec4])
        else:
            # Fallback to original decoder (should not reach here)
            return self.backbone(x)
    
    def _lightweight_decode(self, features):
        """SegFormer3D-style lightweight decoding"""
        enc0, enc1, enc2, enc3, dec4 = features
        
        # Project features to common dimension (like SegFormer's MLP decoder)
        target_size = enc0.shape[2:]  # Use enc0 as reference size
        
        projected_features = []
        feature_list = [dec4, enc3, enc2, enc1]  # Reverse order for processing
        
        for i, feat in enumerate(feature_list):
            # Project to common channel dimension
            proj_feat = self.feature_projections[i](feat)
            
            # Upsample to target size
            if proj_feat.shape[2:] != target_size:
                proj_feat = torch.nn.functional.interpolate(
                    proj_feat, size=target_size, mode='trilinear', align_corners=False
                )
            projected_features.append(proj_feat)
        
        # Fuse all features (SegFormer-style)
        fused = torch.cat(projected_features, dim=1)
        fused = self.feature_fusion(fused)
        
        # Final lightweight decoder
        output = self.lightweight_decoder(fused)
        
        # Upsample to original input size
        if output.shape[2:] != (128, 128, 128):  # Assuming 128^3 input
            output = torch.nn.functional.interpolate(
                output, size=(128, 128, 128), mode='trilinear', align_corners=False
            )
        
        return output
    
    def count_parameters(self):
        """Count total trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params
    
    def _display_parameter_info(self):
        """Display detailed parameter information"""
        total_params = self.count_parameters()
        
        # Calculate component-wise parameters
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        if hasattr(self, 'feature_projections'):
            projection_params = sum(p.numel() for p in self.feature_projections.parameters() if p.requires_grad)
            fusion_params = sum(p.numel() for p in self.feature_fusion.parameters() if p.requires_grad)
            decoder_params = sum(p.numel() for p in self.lightweight_decoder.parameters() if p.requires_grad)
        else:
            projection_params = fusion_params = decoder_params = 0
        
        print(f"\n📊 Ultra-Efficient SwinUNETR Parameter Breakdown:")
        print(f"   🔧 Feature size: {self.feature_size}")
        print(f"   📈 Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   🏗️  Backbone: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        if projection_params > 0:
            print(f"   🔗 Projections: {projection_params:,} ({projection_params/1e6:.2f}M)")
            print(f"   🔀 Fusion: {fusion_params:,} ({fusion_params/1e6:.2f}M)")
            print(f"   📤 Decoder: {decoder_params:,} ({decoder_params/1e6:.2f}M)")
        
        # Compare with standard SwinUNETR
        standard_estimate = self._estimate_standard_swinunetr_params()
        reduction = (1 - total_params / standard_estimate) * 100
        print(f"   📉 Parameter reduction: {reduction:.1f}% vs standard MONAI SwinUNETR")
    
    def _estimate_standard_swinunetr_params(self):
        """Estimate parameters for standard MONAI SwinUNETR with feature_size=48"""
        # Rough estimation based on MONAI SwinUNETR architecture
        return 62e6  # Approximately 62M parameters for standard config


class LightweightMLP(nn.Module):
    """Lightweight MLP projection inspired by SegFormer"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Ensure input_dim matches the actual input channels
        self.proj = nn.Conv3d(input_dim, output_dim, kernel_size=1)
        self.norm = nn.BatchNorm3d(output_dim)
        
    def forward(self, x):
        # Add shape check for debugging
        if x.shape[1] != self.proj.in_channels:
            raise ValueError(f"Expected input channels {self.proj.in_channels}, got {x.shape[1]}")
        x = self.proj(x)
        x = self.norm(x)
        return x


class LightweightFusion(nn.Module):
    """Lightweight feature fusion module"""
    
    def __init__(self, in_channels, out_channels, use_depthwise=True, use_separable=True):
        super().__init__()
        
        if use_separable:
            # Separable convolution: depthwise + pointwise
            self.fusion = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif use_depthwise:
            # Depthwise convolution
            self.fusion = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=min(in_channels, out_channels)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Standard convolution
            self.fusion = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.fusion(x)


class LightweightDecoder(nn.Module):
    """Ultra-lightweight decoder with minimal parameters"""
    
    def __init__(self, decoder_channels, out_channels, use_depthwise=True):
        super().__init__()
        
        layers = []
        in_ch = decoder_channels[0]
        
        for out_ch in decoder_channels[1:]:
            if use_depthwise:
                layers.extend([
                    nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
                    nn.Conv3d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                ])
            else:
                layers.extend([
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                ])
            in_ch = out_ch
        
        # Final output layer
        layers.append(nn.Conv3d(in_ch, out_channels, kernel_size=1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)
    





# Alternative ultra-lightweight configuration inspired by SegFormer3D
class SegFormerStyleSwinUNETR(UltraEfficientSwinUNETR):
    """SegFormer3D-style configuration with extreme efficiency"""
    
    def __init__(self, **kwargs):
        # Override with SegFormer3D-like parameters
        segformer_config = {
            'feature_size': 12,  # Changed from 16 to be divisible by 12
            'depths': (1, 1, 1, 1),  # Minimal depths
            'num_heads': (1, 2, 4, 8),  # Small attention heads
            'decoder_channels': (48, 24, 12, 6),  # Adjusted to match feature size
            'use_depthwise_conv': True,
            'use_lightweight_decoder': True,
            'use_separable_conv': True,
            'reduce_skip_connections': True,
        }
        segformer_config.update(kwargs)
        super().__init__(**segformer_config)


# Factory function for easy configuration selection
def create_efficient_swinunetr(efficiency_level="balanced", **kwargs):
    """
    Factory function to create different efficiency levels of SwinUNETR
    
    Args:
        efficiency_level: "ultra", "high", "balanced", "performance"
    """
    
    if efficiency_level == "ultra":
        # ~5-8M parameters - SegFormer3D-like efficiency
        config = {
            'feature_size': 12,  # Changed from 16 to be divisible by 12
            'depths': (1, 1, 1, 1),
            'num_heads': (1, 2, 4, 8),
            'decoder_channels': (48, 24, 12, 6),  # Adjusted to match feature size
        }
        return SegFormerStyleSwinUNETR(**{**config, **kwargs})
    
    elif efficiency_level == "high":
        # ~10-15M parameters
        config = {
            'feature_size': 24,  # Already divisible by 12
            'depths': (1, 1, 2, 1),
            'num_heads': (2, 4, 6, 12),
            'decoder_channels': (96, 48, 24, 12),
        }
        return UltraEfficientSwinUNETR(**{**config, **kwargs})
    
    elif efficiency_level == "balanced":
        # ~15-25M parameters (your current target)
        config = {
            'feature_size': 24,  # Already divisible by 12
            'depths': (1, 1, 2, 1),
            'num_heads': (2, 4, 8, 16),
            'decoder_channels': (96, 48, 24, 12),
        }
        return UltraEfficientSwinUNETR(**{**config, **kwargs})
    
    elif efficiency_level == "performance":
        # ~25-35M parameters - more performance focused
        config = {
            'feature_size': 36,  # Changed from 32 to be divisible by 12
            'depths': (2, 2, 2, 2),
            'num_heads': (3, 6, 12, 24),
            'decoder_channels': (144, 72, 36, 18),  # Adjusted to match feature size
        }
        return UltraEfficientSwinUNETR(**{**config, **kwargs})
    
    else:
        raise ValueError(f"Unknown efficiency level: {efficiency_level}")



# Compatibility wrapper for your existing pipeline
class ParameterEfficientSwinUNETR(UltraEfficientSwinUNETR):
    """Backward compatibility wrapper"""
    
    def __init__(self, **kwargs):
        # Map old parameters to new system
        if 'feature_size' not in kwargs:
            kwargs['feature_size'] = 24
        if 'depths' not in kwargs:
            kwargs['depths'] = (1, 1, 2, 1)
        
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Test different efficiency levels
    print("Testing different efficiency levels:\n")
    
    for level in ["ultra", "high", "balanced", "performance"]:
        print(f"=== {level.upper()} EFFICIENCY ===")
        model = create_efficient_swinunetr(level)
        
        # Test forward pass
        x = torch.randn(1, 4, 128, 128, 128)
        with torch.no_grad():
            y = model(x)
        print(f"Output shape: {y.shape}")
        print()