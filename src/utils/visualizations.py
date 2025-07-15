#!/usr/bin/env python3
"""
Comprehensive visualization module for SwinUNETR V2 brain tumor segmentation model.
Provides multiple visualization techniques including:
- GradCAM and GradCAM++
- Occlusion sensitivity
- Attention maps
- Tensorboard visualizations with animated GIFs
- Heatmaps and saliency maps
- 3D volume visualization
- Class activation maps

---
### Note on Memory Usage for Visualizations
Due to the high memory requirements of transformer-based 3D models, some visualizations are computed on patches or slices of the input volume to prevent out-of-memory errors.
---

Usage:
    python visualizations.py --checkpoint_path /path/to/checkpoint.pth --image_path /path/to/image
    python visualizations.py --help
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from monai.networks.nets import SwinUNETR
from monai.visualize import (
    GradCAM, GradCAMpp, OcclusionSensitivity,
    blend_images, matshow3d
)
from monai.visualize.img2tensorboard import (
    add_animated_gif, plot_2d_or_3d_image, make_animated_gif_summary
)
from monai.visualize.gradient_based import (
    VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad
)
import torch.nn.functional as F
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, NormalizeIntensityd, Resized
)
from monai.data import Dataset, DataLoader
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

def load_nifti_as_tensor(nii_path, target_shape=(96, 96, 96)):
    img = nib.load(nii_path)
    data = img.get_fdata()
    # If single channel, repeat to 4 channels (for demo)
    if data.ndim == 3:
        data = np.stack([data]*4, axis=0)  # (4, D, H, W)
    elif data.ndim == 4:
        # If already 4 channels, just transpose if needed
        if data.shape[0] != 4:
            data = np.transpose(data, (3, 0, 1, 2))  # (4, D, H, W)
    # Normalize each channel
    for i in range(data.shape[0]):
        d = data[i]
        data[i] = (d - d.mean()) / (d.std() + 1e-8)
    # Resize to target shape
    tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, 4, D, H, W)
    tensor = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
    return tensor

def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=["trilinear", "nearest"]),
    ])

def build_model():
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True,
        spatial_dims=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )

def load_weights(model, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval().to(device), device

def normalize(cam):
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)

def resize_cam(cam, target_shape):
    if cam.ndim == 4:  # [B, H, W, D]
        cam = cam.unsqueeze(1)  # -> [B, 1, H, W, D]
    elif cam.ndim == 3:  # [H, W, D]
        cam = cam[None, None]   # -> [1, 1, H, W, D]
    cam = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
    return normalize(cam)

class ComprehensiveVisualizer:
    """Comprehensive visualization class for SwinUNETR model interpretability."""
    
    def __init__(self, model, device, output_dir="./visualizations"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup tensorboard writer
        self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
        
        # Define class names and colors
        self.class_names = ["Background", "Whole Tumor", "Tumor Core", "Enhancing Tumor"]
        self.colors = ["black", "red", "green", "blue"]
        
    def create_gradcam_visualizations(self, image, target_layers=None, class_idx=None):
        """Create GradCAM and GradCAM++ visualizations."""
        if target_layers is None:
            target_layers = ["encoder1.layer.norm3", "encoder2.layer.norm3", "encoder3.layer.norm3"]
        
        results = {}
        
        for layer in target_layers:
            print(f"Processing GradCAM for layer: {layer}")
            
            # GradCAM
            gradcam = GradCAM(nn_module=self.model, target_layers=layer)
            cam_raw = gradcam(x=image, class_idx=class_idx)
            cam = resize_cam(cam_raw, image.shape[2:])
            results[f"gradcam_{layer}"] = cam
            
            # GradCAM++
            gradcam_pp = GradCAMpp(nn_module=self.model, target_layers=layer)
            cam_pp_raw = gradcam_pp(x=image, class_idx=class_idx)
            cam_pp = resize_cam(cam_pp_raw, image.shape[2:])
            results[f"gradcam_pp_{layer}"] = cam_pp
            
        return results
    
    def create_gradient_visualizations(self, image, class_idx=None):
        """Create gradient-based visualizations."""
        results = {}
        
        # Vanilla Gradient
        vanilla_grad = VanillaGrad(self.model)
        vanilla_result = vanilla_grad(image, class_idx=class_idx)
        results["vanilla_grad"] = vanilla_result
        
        # Smooth Gradient
        smooth_grad = SmoothGrad(self.model, n_samples=10, stdev_spread=0.1)
        smooth_result = smooth_grad(image, class_idx=class_idx)
        results["smooth_grad"] = smooth_result
        
        # Guided Backprop
        guided_grad = GuidedBackpropGrad(self.model)
        guided_result = guided_grad(image, class_idx=class_idx)
        results["guided_grad"] = guided_result
        
        # Guided Backprop + Smooth
        guided_smooth = GuidedBackpropSmoothGrad(self.model, n_samples=10, stdev_spread=0.1)
        guided_smooth_result = guided_smooth(image, class_idx=class_idx)
        results["guided_smooth"] = guided_smooth_result
        
        return results
    
    def create_occlusion_sensitivity(self, image, class_idx=None):
        """Create occlusion sensitivity maps."""
        print("Computing occlusion sensitivity...")
        
        # Use smaller mask size for 3D volumes to avoid memory issues
        occ_sens = OcclusionSensitivity(
            nn_module=self.model,
            mask_size=8,
            n_batch=4,
            mode="gaussian",
            overlap=0.5
        )
        
        occ_map, most_probable_class = occ_sens(image, class_idx=class_idx)
        
        return {
            "occlusion_map": occ_map,
            "most_probable_class": most_probable_class
        }
    
    def create_attention_heatmaps(self, image, predictions=None):
        """Create attention heatmaps from model predictions."""
        with torch.no_grad():
            if predictions is None:
                predictions = self.model(image)
            
            # Apply softmax to get probabilities
            probs = F.softmax(predictions, dim=1)
            
            # Create heatmaps for each class
            heatmaps = {}
            for i, class_name in enumerate(self.class_names):
                if i < probs.shape[1]:
                    heatmap = probs[0, i].cpu().numpy()
                    heatmaps[class_name] = heatmap
                    
            return heatmaps
    
    def create_3d_volume_plots(self, image, cam_maps, predictions=None):
        """Create 3D volume visualizations."""
        input_np = image[0].cpu().numpy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Plot original image (T2 channel)
        plt.subplot(2, 3, 1)
        matshow3d(input_np[3], fig=fig, title="Original T2 Volume", 
                 figsize=(8, 8), frames_per_row=8)
        
        # Plot CAM overlays
        for i, (name, cam) in enumerate(cam_maps.items()):
            if i >= 5:  # Limit to 5 CAMs
                break
            plt.subplot(2, 3, i + 2)
            cam_np = cam[0].cpu().numpy()
            matshow3d(cam_np, fig=fig, title=f"{name} CAM", 
                     figsize=(8, 8), frames_per_row=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "3d_volume_plots.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_tensorboard_visualizations(self, image, cam_maps, predictions=None, step=0):
        """Create tensorboard visualizations with animated GIFs."""
        input_np = image[0].cpu().numpy()
        
        # Log original image as animated GIF
        for i, modality in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            if i < input_np.shape[0]:
                # Convert to CHWD format for tensorboard
                vol_data = input_np[i:i+1]  # [1, H, W, D]
                vol_tensor = torch.from_numpy(vol_data).unsqueeze(0)  # [1, 1, H, W, D]
                
                # Plot 3D volume as animated GIF
                plot_2d_or_3d_image(
                    vol_tensor, step, self.writer, 
                    tag=f"input/{modality}", max_channels=1
                )
        
        # Log CAM maps as animated GIFs
        for name, cam in cam_maps.items():
            cam_tensor = cam.unsqueeze(1)  # Add channel dimension
            plot_2d_or_3d_image(
                cam_tensor, step, self.writer,
                tag=f"cam/{name}", max_channels=1
            )
        
        # Log predictions if available
        if predictions is not None:
            pred_probs = F.softmax(predictions, dim=1)
            for i, class_name in enumerate(self.class_names):
                if i < pred_probs.shape[1]:
                    class_prob = pred_probs[:, i:i+1]  # [1, 1, H, W, D]
                    plot_2d_or_3d_image(
                        class_prob, step, self.writer,
                        tag=f"predictions/{class_name}", max_channels=1
                    )
    
    def create_heatmap_overlays(self, image, cam_maps, slice_idx=None):
        """Create heatmap overlays for different slices."""
        input_np = image[0].cpu().numpy()
        
        if slice_idx is None:
            slice_idx = input_np.shape[2] // 2  # Middle slice
        
        n_cams = len(cam_maps)
        fig, axes = plt.subplots(2, n_cams + 1, figsize=(4 * (n_cams + 1), 8))
        
        # Plot original image
        for i, modality in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            if i < input_np.shape[0]:
                ax = axes[i // 2, 0] if i < 2 else axes[i // 2, 0]
                ax.imshow(input_np[i, :, slice_idx, :], cmap="gray")
                ax.set_title(f"Original {modality}")
                ax.axis("off")
        
        # Plot CAM overlays
        for j, (name, cam) in enumerate(cam_maps.items()):
            cam_np = cam[0].cpu().numpy()
            
            # Axial view
            ax = axes[0, j + 1]
            ax.imshow(input_np[3, :, slice_idx, :], cmap="gray")
            ax.imshow(cam_np[:, slice_idx, :], cmap="jet", alpha=0.5)
            ax.set_title(f"{name} - Axial")
            ax.axis("off")
            
            # Sagittal view
            ax = axes[1, j + 1]
            ax.imshow(input_np[3, :, :, slice_idx], cmap="gray")
            ax.imshow(cam_np[:, :, slice_idx], cmap="jet", alpha=0.5)
            ax.set_title(f"{name} - Sagittal")
            ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"heatmap_overlays_slice_{slice_idx}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_blended_visualizations(self, image, predictions):
        """Create blended image and label visualizations."""
        input_np = image[0].cpu().numpy()
        pred_probs = F.softmax(predictions, dim=1)
        
        # Create predicted labels
        pred_labels = torch.argmax(pred_probs, dim=1, keepdim=True)
        
        # Select middle slice for visualization
        slice_idx = input_np.shape[2] // 2
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Plot each modality with prediction overlay
        for i, modality in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            if i < input_np.shape[0]:
                # Original image
                ax = axes[0, i]
                ax.imshow(input_np[i, :, slice_idx, :], cmap="gray")
                ax.set_title(f"Original {modality}")
                ax.axis("off")
                
                # Blended image
                ax = axes[1, i]
                img_slice = input_np[i:i+1, :, slice_idx, :]
                label_slice = pred_labels[0, :, :, slice_idx, :].cpu().numpy()
                
                # Convert to format expected by blend_images
                img_tensor = torch.from_numpy(img_slice).float()
                label_tensor = torch.from_numpy(label_slice).float()
                
                try:
                    blended = blend_images(img_tensor, label_tensor, alpha=0.5)
                    ax.imshow(blended.numpy()[0], cmap="gray")
                    ax.set_title(f"Blended {modality}")
                    ax.axis("off")
                except Exception as e:
                    print(f"Error creating blended image for {modality}: {e}")
                    ax.imshow(img_slice[0], cmap="gray")
                    ax.set_title(f"Original {modality}")
                    ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "blended_visualizations.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, image, save_path=None):
        """Create a comprehensive summary report with all visualizations."""
        print("Creating comprehensive visualization report...")
        
        # Get model predictions
        with torch.no_grad():
            predictions = self.model(image)
            pred_probs = F.softmax(predictions, dim=1)
        
        # Create all visualizations
        print("1. Creating GradCAM visualizations...")
        cam_maps = self.create_gradcam_visualizations(image, class_idx=1)
        
        print("2. Creating gradient-based visualizations...")
        grad_maps = self.create_gradient_visualizations(image, class_idx=1)
        
        print("3. Creating occlusion sensitivity...")
        try:
            occ_results = self.create_occlusion_sensitivity(image, class_idx=1)
            cam_maps.update({"occlusion": occ_results["occlusion_map"]})
        except Exception as e:
            print(f"Occlusion sensitivity failed: {e}")
        
        print("4. Creating attention heatmaps...")
        attention_maps = self.create_attention_heatmaps(image, predictions)
        
        print("5. Creating 3D volume plots...")
        self.create_3d_volume_plots(image, cam_maps, predictions)
        
        print("6. Creating tensorboard visualizations...")
        self.create_tensorboard_visualizations(image, cam_maps, predictions)
        
        print("7. Creating heatmap overlays...")
        self.create_heatmap_overlays(image, cam_maps)
        
        print("8. Creating blended visualizations...")
        self.create_blended_visualizations(image, predictions)
        
        # Create summary statistics
        self.create_summary_statistics(pred_probs)
        
        print(f"All visualizations saved to: {self.output_dir}")
        self.writer.close()
        
        return {
            "cam_maps": cam_maps,
            "grad_maps": grad_maps,
            "attention_maps": attention_maps,
            "predictions": predictions
        }
    
    def create_summary_statistics(self, predictions):
        """Create summary statistics and plots."""
        pred_probs = predictions[0].cpu().numpy()
        
        # Class probability distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram of probabilities for each class
        for i, (class_name, color) in enumerate(zip(self.class_names, self.colors)):
            if i < pred_probs.shape[0]:
                ax = axes[i // 2, i % 2]
                ax.hist(pred_probs[i].flatten(), bins=50, alpha=0.7, color=color, density=True)
                ax.set_title(f"{class_name} Probability Distribution")
                ax.set_xlabel("Probability")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "probability_distributions.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary statistics
        stats = {}
        for i, class_name in enumerate(self.class_names):
            if i < pred_probs.shape[0]:
                stats[class_name] = {
                    "mean": float(pred_probs[i].mean()),
                    "std": float(pred_probs[i].std()),
                    "min": float(pred_probs[i].min()),
                    "max": float(pred_probs[i].max()),
                    "median": float(np.median(pred_probs[i]))
                }
        
        # Save statistics
        import json
        with open(os.path.join(self.output_dir, "summary_statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        return stats

def show_cam_overlay(image, cam, title, channel_idx=3, channel_name="T2"):
    mid = image.shape[2] // 2
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[channel_idx, :, mid, :], cmap="gray")
    plt.title(f"Original - {channel_name}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image[channel_idx, :, mid, :], cmap="gray")
    plt.imshow(cam[0, :, mid, :], cmap="jet_r", alpha=0.5)
    plt.title(f"Grad-CAM Overlay - {channel_name}")
    plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive SwinUNETR Visualization Suite")
    parser.add_argument("--image_path", type=str, required=True, 
                       help="Path to a preprocessed .nii, .nii.gz, .pt or .npy image file (shape [1, 4, 96, 96, 96])")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--target_class", type=int, default=1, 
                       help="Target class index for visualizations")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--gradcam_layers", type=str, nargs="+", 
                       default=["encoder1.layer.norm3", "encoder2.layer.norm3", "encoder3.layer.norm3"],
                       help="Layers for GradCAM visualization")
    parser.add_argument("--quick_mode", action="store_true",
                       help="Run only essential visualizations (faster)")
    parser.add_argument("--skip_occlusion", action="store_true",
                       help="Skip occlusion sensitivity (memory intensive)")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Enable tensorboard visualizations")
    args = parser.parse_args()

    print(f"Loading image from: {args.image_path}")
    # Load image
    if args.image_path.endswith(".nii") or args.image_path.endswith(".nii.gz"):
        image = load_nifti_as_tensor(args.image_path, target_shape=(96, 96, 96))
    elif args.image_path.endswith(".pt"):
        image = torch.load(args.image_path)
    elif args.image_path.endswith(".npy"):
        image = torch.from_numpy(np.load(args.image_path))
    else:
        raise ValueError("Unsupported image format. Use .nii, .nii.gz, .pt or .npy.")
    
    if image.ndim == 4:
        image = image.unsqueeze(0)  # [1, 4, 96, 96, 96]

    print(f"Image shape: {image.shape}")
    print(f"Loading model from: {args.checkpoint_path}")
    
    # Model
    model, device = load_weights(build_model(), args.checkpoint_path)
    image = image.to(device)

    print(f"Model loaded successfully on device: {device}")
    print(f"Creating visualizations in: {args.output_dir}")
    
    # Create comprehensive visualizer
    visualizer = ComprehensiveVisualizer(model, device, args.output_dir)
    
    if args.quick_mode:
        print("Running in quick mode - generating essential visualizations only")
        # Quick mode - just GradCAM and basic visualizations
        cam_maps = visualizer.create_gradcam_visualizations(
            image, target_layers=args.gradcam_layers[:2], class_idx=args.target_class
        )
        visualizer.create_heatmap_overlays(image, cam_maps)
        
        # Basic prediction visualization
        with torch.no_grad():
            predictions = model(image)
        visualizer.create_blended_visualizations(image, predictions)
        
    else:
        print("Running comprehensive visualization suite...")
        # Comprehensive mode - all visualizations
        results = visualizer.create_summary_report(image)
        
        # Traditional GradCAM overlay for backward compatibility
        if args.gradcam_layers:
            gradcam = GradCAM(nn_module=model, target_layers=args.gradcam_layers[0])
            cam_raw = gradcam(x=image, class_idx=args.target_class)
            cam = resize_cam(cam_raw, image.shape[2:])
            
            input_np = image[0].cpu().numpy()
            cam_np = cam[0].cpu().numpy()
            show_cam_overlay(input_np, cam_np, f'Traditional Grad-CAM: class {args.target_class}')
    
    print("Visualization generation complete!")
    print(f"Results saved to: {args.output_dir}")
    
    if args.tensorboard:
        print(f"Tensorboard logs saved to: {os.path.join(args.output_dir, 'tensorboard')}")
        print("To view tensorboard: tensorboard --logdir ./visualizations/tensorboard")


def create_kaggle_visualization_cell():
    """
    Create a Kaggle-compatible visualization cell that can be copied and pasted.
    This function returns the code as a string that can be executed in a Kaggle notebook.
    """
    
    kaggle_code = '''
# Comprehensive SwinUNETR Visualization for Kaggle
# Copy and paste this code into a Kaggle notebook cell

import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.visualize import GradCAM, GradCAMpp, OcclusionSensitivity, blend_images, matshow3d
from monai.visualize.gradient_based import VanillaGrad, SmoothGrad
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings("ignore")

# Load your model and image (replace with your actual paths)
# model = your_trained_model
# image = your_input_image  # shape: [1, 4, 96, 96, 96]

def comprehensive_visualization(model, image, output_dir="./visualizations"):
    """Generate comprehensive visualizations for brain tumor segmentation"""
    
    device = next(model.parameters()).device
    image = image.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧠 Generating comprehensive brain tumor visualization suite...")
    
    # 1. Model predictions
    with torch.no_grad():
        predictions = model(image)
        pred_probs = F.softmax(predictions, dim=1)
    
    print("📊 Model predictions computed")
    
    # 2. GradCAM visualizations
    print("🔥 Creating GradCAM visualizations...")
    target_layers = ["encoder1.layer.norm3", "encoder2.layer.norm3", "encoder3.layer.norm3"]
    cam_maps = {}
    
    for layer in target_layers:
        try:
            # GradCAM
            gradcam = GradCAM(nn_module=model, target_layers=layer)
            cam_raw = gradcam(x=image, class_idx=1)  # Whole tumor class
            cam = F.interpolate(cam_raw.unsqueeze(1), size=image.shape[2:], 
                              mode="trilinear", align_corners=False)
            cam_maps[f"gradcam_{layer}"] = cam.squeeze(1)
            
            # GradCAM++
            gradcam_pp = GradCAMpp(nn_module=model, target_layers=layer)
            cam_pp_raw = gradcam_pp(x=image, class_idx=1)
            cam_pp = F.interpolate(cam_pp_raw.unsqueeze(1), size=image.shape[2:], 
                                 mode="trilinear", align_corners=False)
            cam_maps[f"gradcam_pp_{layer}"] = cam_pp.squeeze(1)
            
        except Exception as e:
            print(f"❌ Failed to create CAM for {layer}: {e}")
    
    # 3. Gradient-based visualizations
    print("🎯 Creating gradient-based visualizations...")
    try:
        vanilla_grad = VanillaGrad(model)
        vanilla_result = vanilla_grad(image, class_idx=1)
        cam_maps["vanilla_grad"] = vanilla_result
        
        smooth_grad = SmoothGrad(model, n_samples=10, stdev_spread=0.1)
        smooth_result = smooth_grad(image, class_idx=1)
        cam_maps["smooth_grad"] = smooth_result
    except Exception as e:
        print(f"❌ Gradient visualizations failed: {e}")
    
    # 4. Multi-slice heatmap overlays
    print("🌈 Creating multi-slice heatmap overlays...")
    input_np = image[0].cpu().numpy()
    
    # Create comprehensive multi-slice visualization
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    
    # Different slice positions
    slice_positions = [
        input_np.shape[2] // 4,      # 25%
        input_np.shape[2] // 2,      # 50% (middle)
        3 * input_np.shape[2] // 4,  # 75%
    ]
    
    modalities = ["T1", "T1ce", "T2", "FLAIR"]
    
    for row, modality_idx in enumerate(modalities):
        if modality_idx < input_np.shape[0]:
            # Original images at different slices
            for col, slice_idx in enumerate(slice_positions):
                ax = axes[row, col]
                ax.imshow(input_np[modality_idx, :, slice_idx, :], cmap="gray")
                ax.set_title(f"{modalities[modality_idx]} - Slice {slice_idx}")
                ax.axis("off")
            
            # CAM overlays
            for col, (cam_name, cam_tensor) in enumerate(list(cam_maps.items())[:3]):
                ax = axes[row, col + 3]
                cam_np = cam_tensor[0].cpu().numpy()
                slice_idx = slice_positions[1]  # Middle slice
                
                ax.imshow(input_np[modality_idx, :, slice_idx, :], cmap="gray")
                ax.imshow(cam_np[:, slice_idx, :], cmap="jet", alpha=0.6)
                ax.set_title(f"{modalities[modality_idx]} + {cam_name}")
                ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_heatmaps.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 3D Volume visualization
    print("📦 Creating 3D volume plots...")
    fig = plt.figure(figsize=(20, 12))
    
    # Original volume (T2)
    plt.subplot(2, 3, 1)
    matshow3d(input_np[2], title="Original T2 Volume", figsize=(6, 6), frames_per_row=8)
    
    # CAM volumes
    for i, (name, cam) in enumerate(list(cam_maps.items())[:5]):
        plt.subplot(2, 3, i + 2)
        cam_np = cam[0].cpu().numpy()
        matshow3d(cam_np, title=f"{name}", figsize=(6, 6), frames_per_row=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3d_volumes.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Prediction probability maps
    print("🎲 Creating prediction probability maps...")
    class_names = ["Background", "Whole Tumor", "Tumor Core", "Enhancing Tumor"]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    slice_idx = input_np.shape[2] // 2
    
    for i, class_name in enumerate(class_names):
        if i < pred_probs.shape[1]:
            # Axial view
            ax = axes[0, i]
            ax.imshow(input_np[2, :, slice_idx, :], cmap="gray")
            prob_map = pred_probs[0, i, :, slice_idx, :].cpu().numpy()
            ax.imshow(prob_map, cmap="hot", alpha=0.6)
            ax.set_title(f"{class_name} - Axial")
            ax.axis("off")
            
            # Sagittal view
            ax = axes[1, i]
            ax.imshow(input_np[2, :, :, slice_idx], cmap="gray")
            prob_map_sag = pred_probs[0, i, :, :, slice_idx].cpu().numpy()
            ax.imshow(prob_map_sag, cmap="hot", alpha=0.6)
            ax.set_title(f"{class_name} - Sagittal")
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_probabilities.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Summary statistics
    print("📈 Creating summary statistics...")
    pred_np = pred_probs[0].cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ["black", "red", "green", "blue"]
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if i < pred_np.shape[0]:
            ax = axes[i // 2, i % 2]
            ax.hist(pred_np[i].flatten(), bins=50, alpha=0.7, color=color, density=True)
            ax.set_title(f"{class_name} Probability Distribution")
            ax.set_xlabel("Probability")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = pred_np[i].mean()
            std_val = pred_np[i].std()
            ax.text(0.7, 0.8, f"Mean: {mean_val:.3f}\\nStd: {std_val:.3f}", 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probability_distributions.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive visualization complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return {
        "cam_maps": cam_maps,
        "predictions": predictions,
        "pred_probs": pred_probs
    }

# Example usage (uncomment and modify for your specific case):
# results = comprehensive_visualization(model, image)
    '''
    
    return kaggle_code


if __name__ == "__main__":
    main()