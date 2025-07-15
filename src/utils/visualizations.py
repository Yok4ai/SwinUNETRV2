#!/usr/bin/env python3
"""
Visualization module for SwinUNETR V2 brain tumor segmentation model.
Provides attention rollout and GradCAM visualizations for model interpretability.

---
### Note on Memory Usage for Visualizations
Due to the high memory requirements of transformer-based 3D models, GradCAM and attention rollout visualizations are computed on a small patch or slice of the input volume, not the entire scan. This allows for meaningful interpretability without running out of GPU memory.
---

Usage:
    python visualizations.py --checkpoint_path /path/to/checkpoint.pth --sample_data_path /path/to/sample
    python visualizations.py --help
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.data import decollate_batch
from monai.visualize import GradCAM, OcclusionSensitivity
from monai.visualize.utils import blend_images, matshow3d
from monai.inferers import sliding_window_inference

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# from src.models.swinunetrplus import SwinUNETR
from monai.networks.nets import SwinUNETR
from src.models.pipeline import BrainTumorSegmentation
from src.data.augmentations import get_transforms
from src.data.convert_labels import ConvertLabels


class SwinUNETRGradCAM:
    """GradCAM implementation for SwinUNETR using MONAI's built-in GradCAM."""
    
    def __init__(self, model: nn.Module, target_layers: List[str] = ["encoder4"]):
        """
        Initialize GradCAM for SwinUNETR.
        
        Args:
            model: SwinUNETR model
            target_layers: List of target layer names for GradCAM
        """
        self.model = model
        self.model.eval()
        self.target_layers = target_layers
        
        # Initialize MONAI GradCAM for each target layer
        self.gradcams = {}
        for layer_name in target_layers:
            try:
                self.gradcams[layer_name] = GradCAM(nn_module=model, target_layers=layer_name)
            except Exception as e:
                print(f"Warning: Could not create GradCAM for layer {layer_name}: {e}")
                print("Available layers:")
                for name, _ in model.named_modules():
                    print(f"  {name}")
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 0, layer_name: str = None) -> torch.Tensor:
        """
        Generate GradCAM for given input.
        
        Args:
            input_tensor: Input tensor of shape (1, C, D, H, W)
            class_idx: Target class index
            layer_name: Specific layer to use (if None, uses first available)
            
        Returns:
            CAM tensor
        """
        if layer_name is None:
            layer_name = self.target_layers[0]
        
        if layer_name not in self.gradcams:
            print(f"Layer {layer_name} not available. Available layers: {list(self.gradcams.keys())}")
            return torch.zeros_like(input_tensor[0, 0])
        
        try:
            # Generate GradCAM using MONAI
            cam = self.gradcams[layer_name](x=input_tensor, class_idx=class_idx)
            return cam.squeeze().cpu()
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            return torch.zeros_like(input_tensor[0, 0])


class AttentionRollout:
    """Attention rollout implementation for SwinUNETR."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention rollout.
        
        Args:
            model: SwinUNETR model
        """
        self.model = model
        self.model.eval()
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention maps."""
        def attention_hook(module, input, output):
            if hasattr(module, 'attention_weights'):
                self.attention_maps.append(module.attention_weights)
        
        # Register hooks on attention modules
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                module.register_forward_hook(attention_hook)
    
    def generate_rollout(self, input_tensor: torch.Tensor, head_fusion: str = "mean") -> torch.Tensor:
        """
        Generate attention rollout.
        
        Args:
            input_tensor: Input tensor of shape (1, C, D, H, W)
            head_fusion: How to fuse attention heads ("mean", "max", "min")
            
        Returns:
            Rollout attention map
        """
        self.attention_maps = []
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self.attention_maps:
            # Fallback: Use gradient-based attention for transformer blocks
            return self._compute_gradient_attention(input_tensor)
        
        # Compute attention rollout
        rollout = self._compute_rollout(self.attention_maps, head_fusion)
        return rollout
    
    def _compute_gradient_attention(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute gradient-based attention as fallback."""
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Sum all output channels
        target_score = output.sum()
        target_score.backward()
        
        # Use input gradients as attention
        attention = torch.abs(input_tensor.grad).mean(dim=1)  # Average across channels
        attention = attention / (attention.max() + 1e-8)
        
        return attention.squeeze().detach().cpu()
    
    def _compute_rollout(self, attention_maps: List[torch.Tensor], head_fusion: str) -> torch.Tensor:
        """Compute attention rollout from attention maps."""
        if not attention_maps:
            return torch.zeros(1)
        
        # Fuse attention heads
        fused_attentions = []
        for attn_map in attention_maps:
            if head_fusion == "mean":
                fused = attn_map.mean(dim=1)
            elif head_fusion == "max":
                fused = attn_map.max(dim=1)[0]
            elif head_fusion == "min":
                fused = attn_map.min(dim=1)[0]
            else:
                fused = attn_map.mean(dim=1)
            
            fused_attentions.append(fused)
        
        # Roll out attention through layers
        rollout = fused_attentions[0]
        for attn in fused_attentions[1:]:
            rollout = torch.matmul(rollout, attn)
        
        return rollout.detach().cpu()


class SwinUNETRVisualizer:
    """Main visualization class for SwinUNETR."""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", roi_size=(96, 96, 96), sw_batch_size=1, patch_size=(16, 96, 96)):
        """
        Initialize visualizer.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run on
            roi_size: ROI size for sliding window inference
            sw_batch_size: Sliding window batch size
            patch_size: Patch size for GradCAM/attention visualization
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.patch_size = patch_size
        
        # Initialize visualization tools
        target_layers = ["encoder4", "encoder3", "encoder2", "swinViT.layers4.0"]
        self.gradcam = SwinUNETRGradCAM(self.model, target_layers=target_layers)
        self.attention_rollout = AttentionRollout(self.model)
        
        # Set mask_size for occlusion sensitivity to a value that divides patch_size
        mask_size = min(8, *patch_size)
        self.occlusion_sensitivity = OcclusionSensitivity(
            nn_module=self.model,
            mask_size=mask_size,
            n_batch=4,
            mode='gaussian'
        )
        
        # Post-processing transforms
        self.post_transforms = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint."""
        # Create standard MONAI SwinUNETR model
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=3,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            use_v2=True,
        )
        
        # Load weights with flexible loading
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load with strict=False to handle missing keys
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded model from {model_path} (some keys may be missing)")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random weights...")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def load_and_preprocess(self, data_path: str) -> Tuple[torch.Tensor, Dict]:
        """Load and preprocess input data."""
        # Handle different input formats
        if data_path.endswith('.nii') or data_path.endswith('.nii.gz'):
            # Single NIfTI file - need to create 4-channel input for SwinUNETR
            img = nib.load(data_path)
            data = img.get_fdata()
            
            # Convert to tensor and add batch/channel dimensions
            data = torch.from_numpy(data).float()
            if data.dim() == 3:
                # Replicate single channel to 4 channels for SwinUNETR
                data = data.unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0)  # (1, 4, D, H, W)
            elif data.dim() == 4:
                data = data.unsqueeze(0)  # Add batch dim
            
            # Normalize each modality
            for i in range(data.shape[1]):
                data[0, i] = (data[0, i] - data[0, i].mean()) / (data[0, i].std() + 1e-8)
            
            metadata = {"spacing": img.header.get_zooms()[:3], "shape": data.shape}
            
        else:
            # Directory with multiple modalities
            data_dir = Path(data_path)
            if data_dir.is_dir():
                # Look for BraTS format files
                modalities = []
                for suffix in ['t1.nii.gz', 't1c.nii.gz', 't2.nii.gz', 'flair.nii.gz']:
                    file_path = data_dir / f"{data_dir.name}-{suffix}"
                    if not file_path.exists():
                        # Try alternative naming
                        file_path = list(data_dir.glob(f"*{suffix}"))[0] if list(data_dir.glob(f"*{suffix}")) else None
                    
                    if file_path and file_path.exists():
                        img = nib.load(file_path)
                        modalities.append(torch.from_numpy(img.get_fdata()).float())
                
                if len(modalities) == 4:
                    data = torch.stack(modalities, dim=0).unsqueeze(0)  # (1, 4, D, H, W)
                    # Normalize each modality
                    for i in range(4):
                        data[0, i] = (data[0, i] - data[0, i].mean()) / (data[0, i].std() + 1e-8)
                    
                    metadata = {"spacing": img.header.get_zooms()[:3], "shape": data.shape}
                else:
                    raise ValueError(f"Expected 4 modalities, found {len(modalities)}")
            else:
                raise ValueError(f"Data path {data_path} not found")
        
        # Ensure spatial dimensions are divisible by 32 (2^5) for SwinUNETR
        data = self._pad_to_divisible(data, divisor=32)
        
        return data.to(self.device), metadata
    
    def _pad_to_divisible(self, tensor: torch.Tensor, divisor: int = 32) -> torch.Tensor:
        """Pad tensor to ensure spatial dimensions are divisible by divisor."""
        B, C, D, H, W = tensor.shape
        
        # Calculate padding needed
        pad_d = (divisor - D % divisor) % divisor
        pad_h = (divisor - H % divisor) % divisor
        pad_w = (divisor - W % divisor) % divisor
        
        # Pad symmetrically
        tensor = F.pad(tensor, (
            pad_w // 2, pad_w - pad_w // 2,  # W padding
            pad_h // 2, pad_h - pad_h // 2,  # H padding
            pad_d // 2, pad_d - pad_d // 2   # D padding
        ), mode='constant', value=0)
        
        print(f"Padded from ({D}, {H}, {W}) to {tensor.shape[2:]}")
        return tensor
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction using sliding window inference to avoid CUDA OOM."""
        with torch.no_grad():
            try:
                output = sliding_window_inference(
                    input_tensor, 
                    roi_size=self.roi_size, 
                    sw_batch_size=self.sw_batch_size, 
                    predictor=self.model, 
                    overlap=0.5
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("CUDA OOM during sliding window inference. Try reducing roi_size or sw_batch_size.")
                raise
            probabilities = torch.sigmoid(output)
            predictions = self.post_transforms(output)
        return predictions, probabilities
    
    def visualize_slice(self, 
                       input_tensor: torch.Tensor, 
                       predictions: torch.Tensor,
                       gradcam: torch.Tensor,
                       attention: torch.Tensor,
                       slice_idx: int,
                       save_path: str = None) -> None:
        """Visualize a single slice with all visualizations."""
        # Extract slice
        img_slice = input_tensor[0, 0, :, :, slice_idx].cpu().numpy()  # First modality
        pred_slice = predictions[0, :, :, :, slice_idx].cpu().numpy()
        gradcam_slice = gradcam[:, :, slice_idx].cpu().numpy()
        attention_slice = attention[:, :, slice_idx].cpu().numpy() if attention.dim() == 3 else attention.cpu().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(img_slice, cmap='gray')
        axes[0, 0].set_title('Original Image (T1)')
        axes[0, 0].axis('off')
        
        # Predictions overlaid
        axes[0, 1].imshow(img_slice, cmap='gray')
        colors = ['red', 'green', 'blue']
        labels = ['TC', 'WT', 'ET']
        for i in range(3):
            mask = pred_slice[i] > 0.5
            if mask.any():
                axes[0, 1].contour(mask, colors=[colors[i]], linewidths=2)
        axes[0, 1].set_title('Predictions Overlay')
        axes[0, 1].axis('off')
        
        # Individual predictions
        axes[0, 2].imshow(pred_slice.sum(axis=0), cmap='viridis')
        axes[0, 2].set_title('Combined Predictions')
        axes[0, 2].axis('off')
        
        # GradCAM
        axes[1, 0].imshow(img_slice, cmap='gray')
        axes[1, 0].imshow(gradcam_slice, cmap='jet', alpha=0.5)
        axes[1, 0].set_title('GradCAM Overlay')
        axes[1, 0].axis('off')
        
        # Attention rollout
        axes[1, 1].imshow(img_slice, cmap='gray')
        axes[1, 1].imshow(attention_slice, cmap='hot', alpha=0.5)
        axes[1, 1].set_title('Attention Rollout')
        axes[1, 1].axis('off')
        
        # Combined visualization
        axes[1, 2].imshow(img_slice, cmap='gray')
        axes[1, 2].imshow(gradcam_slice, cmap='jet', alpha=0.3)
        axes[1, 2].imshow(attention_slice, cmap='hot', alpha=0.3)
        axes[1, 2].set_title('Combined Visualization')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def generate_all_visualizations(self, 
                                  input_tensor: torch.Tensor, 
                                  output_dir: str = "visualizations") -> Dict:
        """Generate all visualizations for the input."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions
        predictions, probabilities = self.predict(input_tensor)
        
        # --- Extract a center patch for GradCAM/attention to avoid OOM ---
        B, C, D, H, W = input_tensor.shape
        pd, ph, pw = self.patch_size
        d0 = max(0, D//2 - pd//2)
        h0 = max(0, H//2 - ph//2)
        w0 = max(0, W//2 - pw//2)
        d1 = min(D, d0 + pd)
        h1 = min(H, h0 + ph)
        w1 = min(W, w0 + pw)
        center_patch = input_tensor[:, :, d0:d1, h0:h1, w0:w1]
        # Pad patch to ensure all spatial dims are divisible by 32
        def pad_to_divisible(tensor, divisor=32):
            B, C, D, H, W = tensor.shape
            pad_d = (divisor - D % divisor) % divisor
            pad_h = (divisor - H % divisor) % divisor
            pad_w = (divisor - W % divisor) % divisor
            return F.pad(tensor, (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2
            ), mode='constant', value=0)
        center_patch = pad_to_divisible(center_patch, divisor=32)
        
        # Generate visualizations for each class
        results = {}
        for class_idx in range(3):
            class_name = ['TC', 'WT', 'ET'][class_idx]
            
            # GradCAM for different layers (on patch)
            gradcam_maps = {}
            for layer_name in self.gradcam.target_layers:
                if layer_name in self.gradcam.gradcams:
                    gradcam_maps[layer_name] = self.gradcam.generate_cam(
                        center_patch, class_idx, layer_name
                    )
            
            # Attention rollout (on patch)
            attention = self.attention_rollout.generate_rollout(center_patch)
            
            # Occlusion sensitivity (on patch)
            # Only run occlusion if patch is large enough for mask
            mask_size = self.occlusion_sensitivity.mask_size
            _, _, pd, ph, pw = center_patch.shape
            if pd >= mask_size and ph >= mask_size and pw >= mask_size:
                try:
                    occ_map, _ = self.occlusion_sensitivity(center_patch)
                    occlusion = occ_map[0, :, :, :, class_idx].detach().cpu()
                except Exception as e:
                    print(f"Warning: Could not compute occlusion sensitivity for {class_name}: {e}")
                    occlusion = torch.zeros_like(center_patch[0, 0])
            else:
                print(f"Skipping occlusion sensitivity for {class_name}: patch too small for mask_size {mask_size}")
                occlusion = torch.zeros_like(center_patch[0, 0])
            
            results[class_name] = {
                'gradcam_maps': gradcam_maps,
                'attention': attention,
                'occlusion': occlusion,
                'predictions': predictions[0, class_idx, d0:d1, h0:h1, w0:w1],
                'probabilities': probabilities[0, class_idx, d0:d1, h0:h1, w0:w1]
            }
        
        # Save visualizations for middle slices (in patch)
        patch_depth = d1 - d0
        middle_slices = [patch_depth // 4, patch_depth // 2, 3 * patch_depth // 4]
        for slice_idx in middle_slices:
            for class_idx, class_name in enumerate(['TC', 'WT', 'ET']):
                save_path = os.path.join(output_dir, f"{class_name}_patch_slice_{slice_idx}.png")
                self.visualize_slice_comprehensive(
                    center_patch, 
                    predictions[:, :, d0:d1, h0:h1, w0:w1],
                    results[class_name],
                    slice_idx,
                    save_path
                )
        
        # Generate 3D visualizations using MONAI's matshow3d (on patch)
        self.generate_3d_visualizations(center_patch, results, output_dir)
        
        return results
    
    def visualize_slice_comprehensive(self, 
                                    input_tensor: torch.Tensor, 
                                    predictions: torch.Tensor,
                                    class_results: Dict,
                                    slice_idx: int,
                                    save_path: str = None) -> None:
        """Comprehensive visualization with multiple techniques."""
        # Extract slice
        img_slice = input_tensor[0, 0, :, :, slice_idx].detach().cpu().numpy()  # T1 modality
        pred_slice = predictions[0, :, :, :, slice_idx].detach().cpu().numpy()
        
        # Get the first available GradCAM map
        gradcam_slice = None
        for layer_name, gradcam_map in class_results['gradcam_maps'].items():
            if gradcam_map.dim() == 3:
                gradcam_slice = gradcam_map[:, :, slice_idx].detach().cpu().numpy()
                break
        
        attention_slice = class_results['attention'][:, :, slice_idx].detach().cpu().numpy() if class_results['attention'].dim() == 3 else class_results['attention'].detach().cpu().numpy()
        occlusion_slice = class_results['occlusion'][:, :, slice_idx].detach().cpu().numpy() if class_results['occlusion'].dim() == 3 else class_results['occlusion'].detach().cpu().numpy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original image
        axes[0, 0].imshow(img_slice, cmap='gray')
        axes[0, 0].set_title('Original Image (T1)')
        axes[0, 0].axis('off')
        
        # Prediction overlay
        axes[0, 1].imshow(img_slice, cmap='gray')
        if pred_slice.sum() > 0:
            axes[0, 1].contour(pred_slice.sum(axis=0), colors=['red'], linewidths=2)
        axes[0, 1].set_title('Prediction Overlay')
        axes[0, 1].axis('off')
        
        # GradCAM
        axes[0, 2].imshow(img_slice, cmap='gray')
        if gradcam_slice is not None:
            axes[0, 2].imshow(gradcam_slice, cmap='jet', alpha=0.5)
        axes[0, 2].set_title('GradCAM')
        axes[0, 2].axis('off')
        
        # Attention
        axes[0, 3].imshow(img_slice, cmap='gray')
        axes[0, 3].imshow(attention_slice, cmap='hot', alpha=0.5)
        axes[0, 3].set_title('Attention')
        axes[0, 3].axis('off')
        
        # Occlusion sensitivity
        axes[1, 0].imshow(img_slice, cmap='gray')
        axes[1, 0].imshow(occlusion_slice, cmap='coolwarm', alpha=0.5)
        axes[1, 0].set_title('Occlusion Sensitivity')
        axes[1, 0].axis('off')
        
        # Blended visualization using MONAI's blend_images
        try:
            # Convert to CHW format for MONAI
            img_chw = torch.from_numpy(img_slice).unsqueeze(0)
            pred_chw = torch.from_numpy(pred_slice.sum(axis=0)).unsqueeze(0)
            blended = blend_images(img_chw, pred_chw, alpha=0.5)
            axes[1, 1].imshow(blended.squeeze().numpy(), cmap='viridis')
            axes[1, 1].set_title('MONAI Blended')
        except Exception as e:
            axes[1, 1].imshow(img_slice, cmap='gray')
            axes[1, 1].set_title('Blended (Error)')
        axes[1, 1].axis('off')
        
        # Combined visualization
        axes[1, 2].imshow(img_slice, cmap='gray')
        if gradcam_slice is not None:
            axes[1, 2].imshow(gradcam_slice, cmap='jet', alpha=0.3)
        axes[1, 2].imshow(attention_slice, cmap='hot', alpha=0.3)
        axes[1, 2].set_title('Combined')
        axes[1, 2].axis('off')
        
        # Probability heatmap
        prob_slice = class_results['probabilities'][:, :, slice_idx].detach().cpu().numpy()
        axes[1, 3].imshow(prob_slice, cmap='plasma')
        axes[1, 3].set_title('Probability Map')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive visualization to {save_path}")
        
        plt.close()
    
    def generate_3d_visualizations(self, input_tensor: torch.Tensor, results: Dict, output_dir: str) -> None:
        """Generate 3D visualizations using MONAI's matshow3d."""
        for class_name, data in results.items():
            # Create 3D visualization for each visualization type
            for viz_type in ['gradcam_maps', 'attention', 'occlusion']:
                try:
                    if viz_type == 'gradcam_maps':
                        # Use the first available GradCAM map
                        for layer_name, gradcam_map in data['gradcam_maps'].items():
                            if gradcam_map.dim() == 3:
                                fig = plt.figure(figsize=(15, 10))
                                matshow3d(gradcam_map.numpy(), fig=fig, 
                                         title=f'{class_name} GradCAM - {layer_name}')
                                save_path = os.path.join(output_dir, f'{class_name}_gradcam_{layer_name}_3d.png')
                                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                                plt.close()
                                break
                    else:
                        viz_data = data[viz_type]
                        if viz_data.dim() == 3:
                            fig = plt.figure(figsize=(15, 10))
                            matshow3d(viz_data.numpy(), fig=fig, 
                                     title=f'{class_name} {viz_type.capitalize()}')
                            save_path = os.path.join(output_dir, f'{class_name}_{viz_type}_3d.png')
                            plt.savefig(save_path, dpi=300, bbox_inches='tight')
                            plt.close()
                except Exception as e:
                    print(f"Error generating 3D visualization for {class_name} {viz_type}: {e}")
    
    def save_attention_maps(self, results: Dict, output_dir: str) -> None:
        """Save attention maps as NIfTI files."""
        for class_name, data in results.items():
            # Save GradCAM maps
            for layer_name, gradcam_map in data['gradcam_maps'].items():
                if gradcam_map.dim() == 3:
                    gradcam_path = os.path.join(output_dir, f"{class_name}_gradcam_{layer_name}.nii.gz")
                    gradcam_img = nib.Nifti1Image(gradcam_map.numpy(), affine=np.eye(4))
                    nib.save(gradcam_img, gradcam_path)
            
            # Save attention rollout
            attention_path = os.path.join(output_dir, f"{class_name}_attention.nii.gz")
            attention_data = data['attention']
            if attention_data.dim() == 2:
                # If attention is 2D, expand to 3D by repeating the 2D map
                attention_data = attention_data.unsqueeze(-1).repeat(1, 1, 96)  # Assume 96 depth
            attention_img = nib.Nifti1Image(attention_data.numpy(), affine=np.eye(4))
            nib.save(attention_img, attention_path)
            
            # Save occlusion sensitivity
            occlusion_path = os.path.join(output_dir, f"{class_name}_occlusion.nii.gz")
            occlusion_data = data['occlusion']
            if occlusion_data.dim() == 3:
                occlusion_img = nib.Nifti1Image(occlusion_data.numpy(), affine=np.eye(4))
                nib.save(occlusion_img, occlusion_path)
            
            print(f"Saved {class_name} attention maps to {output_dir}")


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="SwinUNETR Visualization Tool")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--sample_data_path", type=str, required=True,
                       help="Path to sample data (directory or single file)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--target_layers", nargs="+", default=["encoder4"],
                       help="Target layers for GradCAM")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    print(f"Initializing visualizer with checkpoint: {args.checkpoint_path}")
    visualizer = SwinUNETRVisualizer(args.checkpoint_path, args.device)
    
    # Load and preprocess data
    print(f"Loading data from: {args.sample_data_path}")
    input_tensor, metadata = visualizer.load_and_preprocess(args.sample_data_path)
    print(f"Data shape: {input_tensor.shape}")
    print(f"Metadata: {metadata}")
    
    # Generate visualizations
    print("Generating visualizations...")
    results = visualizer.generate_all_visualizations(input_tensor, args.output_dir)
    
    # Save attention maps as NIfTI
    print("Saving attention maps...")
    visualizer.save_attention_maps(results, args.output_dir)
    
    print(f"Visualization complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()