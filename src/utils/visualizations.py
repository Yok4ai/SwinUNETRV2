#!/usr/bin/env python3
"""
Simplified visualization module for SwinUNETR V2 brain tumor segmentation model.
Provides basic GradCAM visualization and prediction overlay.

Usage:
    python visualizations.py --checkpoint_path /path/to/checkpoint.pth --image_path /path/to/image
"""

import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.visualize import GradCAM
import nibabel as nib

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

def show_cam_overlay(image, cam, title, channel_idx=3, channel_name="T2", save_path=None):
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
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def load_image(image_path):
    ext = os.path.splitext(image_path)[1]
    if ext in [".nii", ".gz"]:
        img = nib.load(image_path)
        data = img.get_fdata()
        if data.ndim == 4:
            return torch.from_numpy(data).float()
        else:
            raise ValueError("Expected 4D image (C, H, W, D) for .nii/.nii.gz")
    elif ext == ".pt":
        data = torch.load(image_path)
        if isinstance(data, torch.Tensor):
            return data.float()
        elif isinstance(data, dict) and "image" in data:
            return data["image"].float()
        else:
            raise ValueError(".pt file must contain a tensor or dict with 'image' key")
    elif ext == ".npy":
        data = np.load(image_path)
        return torch.from_numpy(data).float()
    else:
        raise ValueError(f"Unsupported image file extension: {ext}")

def run_gradcam(
    image_path,
    checkpoint_path,
    target_class=1,
    output_dir="./visualizations"
):
    model = build_model()
    model, device = load_weights(model, checkpoint_path)
    image = load_image(image_path)
    if image.ndim == 4:
        image = image.unsqueeze(0)  # [1, C, H, W, D]
    image = image.to(device)
    gradcam = GradCAM(nn_module=model, target_layers="encoder1.layer.norm3")
    cam_raw = gradcam(x=image, class_idx=target_class)
    cam = resize_cam(cam_raw, image.shape[2:])
    input_np = image[0].cpu().numpy()
    cam_np = cam[0].cpu().numpy()
    class_names = ["Tumor Core", "Whole Tumor", "Enhancing Tumor"]
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"gradcam_{os.path.basename(image_path)}_class{target_class}.png")
    show_cam_overlay(input_np, cam_np, f"Grad-CAM: {class_names[target_class]}", save_path=save_path)
    print(f"Saved GradCAM visualization to {save_path}")
    return input_np, cam_np

def main():
    parser = argparse.ArgumentParser(description="SwinUNETR GradCAM Visualization CLI (single image)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a preprocessed .nii, .nii.gz, .pt or .npy image file (shape [C, H, W, D] or [1, C, H, W, D])")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--target_class", type=int, default=1, help="Target class index for GradCAM visualization")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory for visualizations")
    args = parser.parse_args()
    run_gradcam(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        target_class=args.target_class,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()