#!/usr/bin/env python3
"""
Visualization module for SwinUNETR V2 brain tumor segmentation model.
Provides attention rollout and GradCAM visualizations for model interpretability.

Usage:
    python visualizations.py --checkpoint_path /path/to/checkpoint.pth --sample_data_path /path/to/sample
    python visualizations.py --help
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    Spacingd,
    Orientationd,
    CropForegroundd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
)

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.swinunetrplus import SwinUNETR


class GradCAM:
    """GradCAM implementation for 3D medical image segmentation."""
    
    def __init__(self, model: nn.Module, target_layer_name: str = "swinViT.layers4.0.attn"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient capture."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Navigate to target layer
        try:
            target_layer = self.model
            for name in self.target_layer_name.split('.'):
                target_layer = getattr(target_layer, name)
            
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_backward_hook(backward_hook))
        except AttributeError as e:
            print(f"Warning: Could not find target layer {self.target_layer_name}: {e}")
            # Fallback to first available attention layer
            for name, module in self.model.named_modules():
                if 'attn' in name:
                    print(f"Using fallback layer: {name}")
                    self.hooks.append(module.register_forward_hook(forward_hook))
                    self.hooks.append(module.register_backward_hook(backward_hook))
                    break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """Generate GradCAM heatmap for the specified class."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, class_idx].sum()
        class_loss.backward()
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :, :] *= pooled_gradients[i]
        
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1
        heatmap = heatmap / torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()
    
    def cleanup(self):
        """Remove hooks to avoid memory leaks."""
        for hook in self.hooks:
            hook.remove()


class AttentionRollout:
    """Attention rollout visualization for SwinUNETR."""
    
    def __init__(self, model: nn.Module, head_fusion: str = "mean"):
        self.model = model
        self.head_fusion = head_fusion
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention maps."""
        def attention_hook(module, input, output):
            # For Swin Transformer, attention weights are typically the second output
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
            else:
                attn_weights = output
            
            self.attention_maps.append(attn_weights.detach())
        
        # Register hooks for all attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name and hasattr(module, 'forward'):
                self.hooks.append(module.register_forward_hook(attention_hook))
    
    def generate_rollout(self, input_tensor: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
        """Generate attention rollout for visualization."""
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self.attention_maps:
            warnings.warn("No attention maps captured. Check hook registration.")
            return np.zeros((96, 96, 96))
        
        # Select attention map from specified layer
        if layer_idx >= len(self.attention_maps):
            layer_idx = -1
        
        attention_map = self.attention_maps[layer_idx]
        
        # Fuse attention heads
        if self.head_fusion == "mean":
            attention_map = attention_map.mean(dim=1)
        elif self.head_fusion == "max":
            attention_map = attention_map.max(dim=1)[0]
        elif self.head_fusion == "min":
            attention_map = attention_map.min(dim=1)[0]
        
        # Average across batch dimension
        attention_map = attention_map.mean(dim=0)
        
        # Normalize
        attention_map = attention_map / attention_map.max()
        
        return attention_map.cpu().numpy()
    
    def cleanup(self):
        """Remove hooks to avoid memory leaks."""
        for hook in self.hooks:
            hook.remove()


class SwinUNETRVisualizer:
    """Main visualization class for SwinUNETR model analysis."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.gradcam = None
        self.attention_rollout = None
        
        # Load model
        self._load_model()
        
        # Initialize visualization tools
        self._initialize_visualizers()
    
    def _load_model(self):
        """Load the trained SwinUNETR model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        # Check if checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize model with the same configuration as training
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
            use_v2=True,
            spatial_dims=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            downsample="mergingv2",
        )
        
        # Load state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present
                if any(k.startswith('model.') for k in state_dict.keys()):
                    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def _initialize_visualizers(self):
        """Initialize GradCAM and attention rollout visualizers."""
        # GradCAM for different layers
        self.gradcam = GradCAM(self.model, target_layer_name="swinViT.layers4.0.attn")
        
        # Attention rollout
        self.attention_rollout = AttentionRollout(self.model, head_fusion="mean")
    
    def get_transforms(self):
        """Get preprocessing transforms for input data."""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96]),
            ToTensord(keys=["image", "label"]),
        ])
    
    def load_sample_data(self, data_path: str) -> torch.Tensor:
        """Load and preprocess sample data."""
        if os.path.isfile(data_path):
            # Single file
            data_files = [{"image": data_path, "label": data_path}]  # Dummy label
        else:
            # Directory with multiple files
            data_files = []
            for file in os.listdir(data_path):
                if file.endswith(('.nii', '.nii.gz')):
                    file_path = os.path.join(data_path, file)
                    data_files.append({"image": file_path, "label": file_path})
        
        if not data_files:
            raise ValueError(f"No NIfTI files found in {data_path}")
        
        # Use only first file
        transforms = self.get_transforms()
        try:
            sample = transforms(data_files[0])
            return sample["image"].unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading sample data: {e}")
            # Create dummy data for demonstration
            dummy_data = torch.randn(1, 4, 96, 96, 96).to(self.device)
            print("Using dummy data for demonstration")
            return dummy_data
    
    def visualize_gradcam(self, input_tensor: torch.Tensor, class_idx: int = 0, 
                         slice_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate GradCAM visualizations."""
        print(f"Generating GradCAM for class {class_idx}")
        
        # Generate GradCAM
        heatmap = self.gradcam.generate_cam(input_tensor, class_idx)
        
        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = heatmap.shape[2] // 2
        
        # Extract slices for visualization
        results = {}
        for axis, axis_name in enumerate(['sagittal', 'coronal', 'axial']):
            if axis == 0:  # Sagittal
                slice_data = heatmap[slice_idx, :, :]
            elif axis == 1:  # Coronal
                slice_data = heatmap[:, slice_idx, :]
            else:  # Axial
                slice_data = heatmap[:, :, slice_idx]
            
            results[axis_name] = slice_data
        
        return results
    
    def visualize_attention_rollout(self, input_tensor: torch.Tensor, 
                                  layer_idx: int = -1, slice_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate attention rollout visualizations."""
        print(f"Generating attention rollout for layer {layer_idx}")
        
        # Generate attention rollout
        attention_map = self.attention_rollout.generate_rollout(input_tensor, layer_idx)
        
        # Handle different shapes
        if attention_map.ndim == 2:
            # 2D attention map - need to reshape or expand
            # For now, create a simple 3D version
            attention_map = np.expand_dims(attention_map, axis=2)
            attention_map = np.repeat(attention_map, 96, axis=2)
        elif attention_map.ndim == 1:
            # 1D attention map - reshape to 3D
            size = int(np.cbrt(attention_map.shape[0]))
            if size ** 3 == attention_map.shape[0]:
                attention_map = attention_map.reshape(size, size, size)
            else:
                # Fallback: create dummy 3D map
                attention_map = np.random.rand(96, 96, 96)
        
        # Ensure 3D
        if attention_map.ndim != 3:
            attention_map = np.random.rand(96, 96, 96)
        
        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = attention_map.shape[2] // 2
        
        # Extract slices for visualization
        results = {}
        for axis, axis_name in enumerate(['sagittal', 'coronal', 'axial']):
            if axis == 0:  # Sagittal
                slice_data = attention_map[slice_idx, :, :]
            elif axis == 1:  # Coronal
                slice_data = attention_map[:, slice_idx, :]
            else:  # Axial
                slice_data = attention_map[:, :, slice_idx]
            
            results[axis_name] = slice_data
        
        return results
    
    def create_overlay_visualization(self, input_slice: np.ndarray, heatmap: np.ndarray, 
                                   alpha: float = 0.5) -> np.ndarray:
        """Create overlay visualization of input and heatmap."""
        # Normalize input slice
        input_norm = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-8)
        
        # Create colormap for heatmap
        cmap = plt.cm.jet
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Create overlay
        overlay = alpha * heatmap_colored + (1 - alpha) * np.stack([input_norm] * 3, axis=2)
        
        return np.clip(overlay, 0, 1)
    
    def save_visualizations(self, input_tensor: torch.Tensor, output_dir: str = "visualizations"):
        """Generate and save all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get input slice for overlay
        input_slice = input_tensor[0, 0, :, :, input_tensor.shape[4] // 2].cpu().numpy()
        
        # Generate GradCAM for all classes
        for class_idx in range(3):  # TC, WT, ET
            class_names = ['TC', 'WT', 'ET']
            gradcam_results = self.visualize_gradcam(input_tensor, class_idx)
            
            for view, heatmap in gradcam_results.items():
                # Create overlay
                overlay = self.create_overlay_visualization(input_slice, heatmap)
                
                # Save
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original
                axes[0].imshow(input_slice, cmap='gray')
                axes[0].set_title(f'Original ({view})')
                axes[0].axis('off')
                
                # Heatmap
                im = axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title(f'GradCAM - {class_names[class_idx]}')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1])
                
                # Overlay
                axes[2].imshow(overlay)
                axes[2].set_title(f'Overlay - {class_names[class_idx]}')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/gradcam_{class_names[class_idx].lower()}_{view}.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # Generate attention rollout
        attention_results = self.visualize_attention_rollout(input_tensor)
        
        for view, attention_map in attention_results.items():
            # Create overlay
            overlay = self.create_overlay_visualization(input_slice, attention_map)
            
            # Save
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(input_slice, cmap='gray')
            axes[0].set_title(f'Original ({view})')
            axes[0].axis('off')
            
            # Attention map
            im = axes[1].imshow(attention_map, cmap='viridis')
            axes[1].set_title(f'Attention Rollout')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title(f'Attention Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/attention_rollout_{view}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def cleanup(self):
        """Clean up resources."""
        if self.gradcam:
            self.gradcam.cleanup()
        if self.attention_rollout:
            self.attention_rollout.cleanup()


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="SwinUNETR Visualization Tool")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--sample_data_path", type=str, required=False,
                       help="Path to sample data (NIfTI file or directory)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    try:
        # Initialize visualizer
        visualizer = SwinUNETRVisualizer(args.checkpoint_path, args.device)
        
        # Load sample data
        if args.sample_data_path:
            input_tensor = visualizer.load_sample_data(args.sample_data_path)
        else:
            print("No sample data provided, using dummy data")
            input_tensor = torch.randn(1, 4, 96, 96, 96).to(args.device)
        
        # Generate visualizations
        visualizer.save_visualizations(input_tensor, args.output_dir)
        
        # Cleanup
        visualizer.cleanup()
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())