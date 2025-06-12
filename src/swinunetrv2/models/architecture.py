# architecture.py - SwinUNETR V2 + SegFormer3D Decoder (Fixed)
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
import math


class SegFormerMLP(nn.Module):
    """SegFormer3D-style MLP projection module (Fixed)"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Use actual input dimension - no artificial constraints
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple linear projection with proper dimension handling
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        print(f"  MLP: {input_dim} -> {output_dim}")
    
    def forward(self, x):
        # x: (B, C, D, H, W) -> (B, D*H*W, C) -> (B, D*H*W, output_dim) -> (B, output_dim, D, H, W)
        B, C, D, H, W = x.shape
        
        # Verify input channels match expected
        if C != self.input_dim:
            raise ValueError(f"Input channels {C} don't match expected {self.input_dim}")
        
        # Flatten spatial dimensions and apply MLP
        x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        x = self.proj(x)                  # (B, D*H*W, output_dim)
        x = self.norm(x)                  # (B, D*H*W, output_dim)
        
        # Reshape back to spatial format
        x = x.transpose(1, 2).reshape(B, self.output_dim, D, H, W)
        
        return x


class HybridSwinUNETR(nn.Module):
    """
    Hybrid model combining:
    - MONAI SwinUNETR V2 backbone (proven, stable)
    - SegFormer3D-style efficient decoder (lightweight)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 36,
        depths: tuple = (1, 1, 2, 1),
        num_heads: tuple = (2, 4, 8, 16),
        use_checkpoint: bool = True,
        use_v2: bool = True,
        spatial_dims: int = 3,
        norm_name: str = "instance",
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        dropout_path_rate: float = 0.1,
        # SegFormer3D decoder parameters
        decoder_embedding_dim: int = 128,
        decoder_dropout: float = 0.0,
        use_segformer_decoder: bool = True,
        **kwargs  # Allow extra parameters
    ):
        super().__init__()
        
        self.feature_size = feature_size
        self.use_segformer_decoder = use_segformer_decoder
        
        # Create SwinUNETR V2 backbone (without final decoder)
        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=feature_size * 4,  # Dummy output, we'll replace decoder
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
            downsample="mergingv2"  # Use V2 merging
        )
        
        if use_segformer_decoder:
            # Replace heavy decoder with SegFormer3D-style lightweight decoder
            self._setup_segformer_decoder(decoder_embedding_dim, out_channels, decoder_dropout)
        
        self._display_parameter_info()
    
    def _setup_segformer_decoder(self, decoder_embedding_dim, out_channels, decoder_dropout):
        """Setup SegFormer3D-style efficient decoder"""
        
        # We'll determine actual feature dimensions dynamically during first forward pass
        # Store parameters for later initialization
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_dropout = decoder_dropout
        self.out_channels = out_channels
        
        # Flag to track if decoder is initialized
        self.decoder_initialized = False
        
        print(f"SegFormer decoder will be initialized dynamically on first forward pass")
        
        # Override forward method to use SegFormer decoder
        self._hook_segformer_forward()
    
    def _initialize_decoder_layers(self, feature_dims):
        """Initialize decoder layers with actual feature dimensions"""
        print(f"Initializing SegFormer decoder with actual dims: {feature_dims}")
        
        # SegFormer3D-style MLP projections for each feature level
        self.mlp_projections = nn.ModuleList([
            SegFormerMLP(feature_dims[4], self.decoder_embedding_dim),  # dec4
            SegFormerMLP(feature_dims[3], self.decoder_embedding_dim),  # enc3
            SegFormerMLP(feature_dims[2], self.decoder_embedding_dim),  # enc2
            SegFormerMLP(feature_dims[1], self.decoder_embedding_dim),  # enc1
        ])
        
        # SegFormer3D-style fusion and final prediction
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(4 * self.decoder_embedding_dim, self.decoder_embedding_dim, 1, bias=False),
            nn.BatchNorm3d(self.decoder_embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout3d(self.decoder_dropout)
        self.final_conv = nn.Conv3d(self.decoder_embedding_dim, self.out_channels, 1)
        
        # Move to same device as backbone
        device = next(self.backbone.parameters()).device
        self.mlp_projections = self.mlp_projections.to(device)
        self.feature_fusion = self.feature_fusion.to(device)
        self.dropout = self.dropout.to(device)
        self.final_conv = self.final_conv.to(device)
        
        self.decoder_initialized = True

    def _hook_segformer_forward(self):
        """Replace SwinUNETR forward with our hybrid approach"""
        original_forward = self.backbone.forward
        
        def hybrid_forward(x):
            # Get SwinUNETR encoder features
            hidden_states_out = self.backbone.swinViT(x, self.backbone.normalize)
            
            # Extract multi-scale features
            enc0 = self.backbone.encoder1(x)                    # 1/4 resolution
            enc1 = self.backbone.encoder2(hidden_states_out[0]) # 1/8 resolution  
            enc2 = self.backbone.encoder3(hidden_states_out[1]) # 1/16 resolution
            enc3 = self.backbone.encoder4(hidden_states_out[2]) # 1/32 resolution
            dec4 = self.backbone.encoder10(hidden_states_out[4]) # 1/32 resolution (deepest)
            
            # Initialize decoder on first forward pass with actual dimensions
            if not self.decoder_initialized:
                feature_dims = [
                    enc0.shape[1],  # actual enc0 channels
                    enc1.shape[1],  # actual enc1 channels
                    enc2.shape[1],  # actual enc2 channels
                    enc3.shape[1],  # actual enc3 channels
                    dec4.shape[1],  # actual dec4 channels
                ]
                print(f"Detected actual feature dimensions: {feature_dims}")
                self._initialize_decoder_layers(feature_dims)
            
            # Apply SegFormer3D decoder
            return self._segformer_decode([enc0, enc1, enc2, enc3, dec4])
        
        self.backbone.forward = hybrid_forward
    
    def forward(self, x):
        if self.use_segformer_decoder:
            # Use our hybrid forward (SwinUNETR encoder + SegFormer decoder)
            return self.backbone(x)
        else:
            # Use standard SwinUNETR
            return self.backbone(x)
    
    def _segformer_decode(self, features):
        """SegFormer3D-style lightweight decoding"""
        enc0, enc1, enc2, enc3, dec4 = features
        
        # Use enc0 as reference size (1/4 of original input)
        target_size = enc0.shape[2:]
        
        # Project all features to common embedding dimension
        projected_features = []
        feature_list = [dec4, enc3, enc2, enc1]  # From deepest to shallowest
        
        for i, feat in enumerate(feature_list):
            # Project to common dimension
            proj_feat = self.mlp_projections[i](feat)
            
            # Upsample to target size (enc0 size)
            if proj_feat.shape[2:] != target_size:
                proj_feat = torch.nn.functional.interpolate(
                    proj_feat, size=target_size, mode='trilinear', align_corners=False
                )
            projected_features.append(proj_feat)
        
        # Concatenate and fuse features (SegFormer3D style)
        fused = torch.cat(projected_features, dim=1)
        fused = self.feature_fusion(fused)
        fused = self.dropout(fused)
        
        # Final prediction
        output = self.final_conv(fused)
        
        # No need for final upsampling since we're already at the right resolution
        return output
    
    def count_parameters(self):
        """Count total trainable parameters"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Add decoder parameters if they exist but aren't part of main parameters yet
        if hasattr(self, 'mlp_projections') and hasattr(self, 'feature_fusion'):
            decoder_total = (
                sum(p.numel() for p in self.mlp_projections.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.feature_fusion.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.final_conv.parameters() if p.requires_grad)
            )
            # Only add if not already counted
            if decoder_total not in [total]:
                total += decoder_total
                
        return total
    
    def _display_parameter_info(self):
        """Display parameter information"""
        # Only show backbone info initially, decoder info will be shown after initialization
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        print(f"\n📊 Hybrid SwinUNETR-SegFormer3D:")
        print(f"   🏗️  Backbone: SwinUNETR V2 (feature_size={self.feature_size})")
        print(f"   🎯 Decoder: SegFormer3D-style MLP decoder (will initialize dynamically)")
        print(f"   🔧 Backbone parameters: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"   📋 Total parameters will be calculated after first forward pass")

    def display_final_parameter_info(self):
        """Display final parameter information after decoder initialization"""
        if not self.decoder_initialized:
            print("⚠️  Decoder not yet initialized. Run a forward pass first.")
            return
            
        total_params = self.count_parameters()
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        decoder_params = total_params - backbone_params
        
        print(f"\n✅ Final Hybrid SwinUNETR-SegFormer3D Statistics:")
        print(f"   📈 Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   🔧 Backbone: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
        print(f"   🔗 SegFormer decoder: {decoder_params:,} ({decoder_params/1e6:.2f}M)")
        
        # Efficiency comparison
        standard_swinunetr = 62e6
        efficiency_gain = (1 - total_params / standard_swinunetr) * 100
        print(f"   📉 Parameter reduction: {efficiency_gain:.1f}% vs standard SwinUNETR")


# Factory function for easy usage
def create_hybrid_swinunetr(
    efficiency_level="balanced",
    decoder_embedding_dim=128,
    use_segformer_decoder=True,
    **kwargs
):
    """
    Create hybrid model with different efficiency levels
    
    Args:
        efficiency_level: "light", "balanced", "performance"
        decoder_embedding_dim: SegFormer decoder embedding dimension
        use_segformer_decoder: Use SegFormer3D-style decoder vs standard
    """
    
    configs = {
        "light": {
            "feature_size": 24,
            "depths": (1, 1, 1, 1),
            "num_heads": (2, 4, 8, 16),
            "decoder_embedding_dim": 96,
        },
        "balanced": {
            "feature_size": 36,
            "depths": (1, 1, 2, 1),
            "num_heads": (2, 4, 8, 16),
            "decoder_embedding_dim": 128,
        },
        "performance": {
            "feature_size": 48,
            "depths": (2, 2, 2, 2),
            "num_heads": (3, 6, 12, 24),
            "decoder_embedding_dim": 192,
        }
    }
    
    if efficiency_level not in configs:
        raise ValueError(f"Unknown efficiency level: {efficiency_level}")
    
    config = configs[efficiency_level]
    config.update(kwargs)
    config["use_segformer_decoder"] = use_segformer_decoder
    config["decoder_embedding_dim"] = decoder_embedding_dim
    
    return HybridSwinUNETR(**config)


# Debug function to check feature dimensions
def debug_feature_dimensions(model, input_shape=(1, 4, 128, 128, 128)):
    """Debug function to check actual feature dimensions"""
    print("\n🔍 Debugging feature dimensions:")
    
    # Create a dummy input
    x = torch.randn(*input_shape)
    
    # Hook to capture feature shapes
    feature_shapes = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                feature_shapes[name] = output.shape
        return hook
    
    # Register hooks for encoder layers
    handles = []
    if hasattr(model.backbone, 'encoder1'):
        handles.append(model.backbone.encoder1.register_forward_hook(get_activation('enc1')))
    if hasattr(model.backbone, 'encoder2'):
        handles.append(model.backbone.encoder2.register_forward_hook(get_activation('enc2')))
    if hasattr(model.backbone, 'encoder3'):
        handles.append(model.backbone.encoder3.register_forward_hook(get_activation('enc3')))
    if hasattr(model.backbone, 'encoder4'):
        handles.append(model.backbone.encoder4.register_forward_hook(get_activation('enc4')))
    if hasattr(model.backbone, 'encoder10'):
        handles.append(model.backbone.encoder10.register_forward_hook(get_activation('enc10')))
    
    # Forward pass
    try:
        with torch.no_grad():
            _ = model(x)
        
        print("Feature shapes:")
        for name, shape in feature_shapes.items():
            print(f"  {name}: {shape}")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        print("Feature shapes captured so far:")
        for name, shape in feature_shapes.items():
            print(f"  {name}: {shape}")
    
    # Clean up hooks
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    # Test the hybrid model
    print("Testing Fixed Hybrid SwinUNETR-SegFormer3D:")
    
    model = create_hybrid_swinunetr(efficiency_level="balanced")
    
    # Debug feature dimensions first
    debug_feature_dimensions(model)
    
    # Test forward pass
    x = torch.randn(1, 4, 128, 128, 128)
    try:
        with torch.no_grad():
            y = model(x)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print("✅ Fixed hybrid model working correctly!")
        
        # Show final parameter statistics
        model.display_final_parameter_info()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This error will help us further debug the model.")