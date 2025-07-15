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
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR
from monai.visualize import GradCAM
import torch.nn.functional as F
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, NormalizeIntensityd, Resized
)
from monai.data import Dataset, DataLoader
import nibabel as nib

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

class AttentionRollout:
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if 'attn' in name.lower():
                self.hooks.append(module.register_forward_hook(self._hook))

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            attn = output[0]
        else:
            attn = output
        self.attentions.append(attn.detach().cpu())

    def rollout(self, x):
        self.attentions = []
        with torch.no_grad():
            _ = self.model(x)
        # Only use the first attention matrix (or pick a specific one)
        if not self.attentions:
            return None
        attn = self.attentions[0]  # shape: [B, num_heads, N, N]
        attn = attn.mean(dim=1)    # mean over heads
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        return attn

def main():
    parser = argparse.ArgumentParser(description="Minimal SwinUNETR GradCAM & Attention Rollout")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a preprocessed .nii, .nii.gz, .pt or .npy image file (shape [1, 4, 96, 96, 96])")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--target_class", type=int, default=1, help="Target class index for GradCAM")
    parser.add_argument("--gradcam_layer", type=str, default="encoder1.layer.norm3", help="Layer for GradCAM")
    args = parser.parse_args()

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

    # Model
    model, device = load_weights(build_model(), args.checkpoint_path)
    image = image.to(device)

    # GradCAM
    gradcam = GradCAM(nn_module=model, target_layers=args.gradcam_layer)
    cam_raw = gradcam(x=image, class_idx=args.target_class)
    cam = resize_cam(cam_raw, image.shape[2:])

    # Attention Rollout
    attn_rollout = AttentionRollout(model)
    attn_map = attn_rollout.rollout(image)

    # Visualization
    input_np = image[0].cpu().numpy()
    cam_np = cam[0].cpu().numpy()
    show_cam_overlay(input_np, cam_np, f'Grad-CAM: class {args.target_class}')

    # Optionally visualize attention rollout (mean over heads, middle token)
    if attn_map is not None:
        attn_np = attn_map[0].mean(0).numpy()
        plt.imshow(attn_np, cmap='hot')
        plt.title('Attention Rollout (mean)')
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    main()