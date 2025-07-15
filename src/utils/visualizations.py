#!/usr/bin/env python3
"""
Simplified visualization module for SwinUNETR V2 brain tumor segmentation model.
Provides basic GradCAM visualization and prediction overlay.

Usage:
    python visualizations.py --checkpoint_path /path/to/checkpoint.pth --image_path /path/to/image
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.visualize import GradCAM, blend_images
import nibabel as nib


def load_nifti_as_tensor(nii_path, target_shape=(96, 96, 96)):
    img = nib.load(nii_path)
    data = img.get_fdata()
    if data.ndim == 3:
        data = np.stack([data]*4, axis=0)  # (4, D, H, W)
    elif data.ndim == 4:
        if data.shape[0] != 4:
            data = np.transpose(data, (3, 0, 1, 2))  # (4, D, H, W)
    for i in range(data.shape[0]):
        d = data[i]
        data[i] = (d - d.mean()) / (d.std() + 1e-8)
    tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, 4, D, H, W)
    tensor = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
    return tensor

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

def resize_cam(cam, target_shape):
    if cam.ndim == 4:  # [B, H, W, D]
        cam = cam.unsqueeze(1)  # -> [B, 1, H, W, D]
    elif cam.ndim == 3:  # [H, W, D]
        cam = cam[None, None]   # -> [1, 1, H, W, D]
    cam = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)
    return cam

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

def create_blended_visualizations(image, predictions, output_dir):
    input_np = image[0].cpu().numpy()
    pred_probs = F.softmax(predictions, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1, keepdim=True)
    slice_idx = input_np.shape[2] // 2
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, modality in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        if i < input_np.shape[0]:
            ax = axes[0, i]
            ax.imshow(input_np[i, :, slice_idx, :], cmap="gray")
            ax.set_title(f"Original {modality}")
            ax.axis("off")
            ax = axes[1, i]
            img_slice = input_np[i:i+1, :, slice_idx, :]
            label_slice = pred_labels[0, :, :, slice_idx, :].cpu().numpy()
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
    plt.savefig(os.path.join(output_dir, "blended_visualizations.png"), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Simple SwinUNETR GradCAM Visualization")
    parser.add_argument("--image_path", type=str, required=True, help="Path to a preprocessed .nii, .nii.gz, .pt or .npy image file (shape [1, 4, 96, 96, 96])")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--target_class", type=int, default=1, help="Target class index for GradCAM visualization")
    args = parser.parse_args()

    print(f"Loading image from: {args.image_path}")
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
    model, device = load_weights(build_model(), args.checkpoint_path)
    image = image.to(device)
    print(f"Model loaded successfully on device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Creating GradCAM visualization in: {args.output_dir}")
    # Only use decoder1 for GradCAM
    gradcam = GradCAM(nn_module=model, target_layers="decoder1")
    cam_raw = gradcam(x=image, class_idx=args.target_class)
    cam = resize_cam(cam_raw, image.shape[2:])
    input_np = image[0].cpu().numpy()
    cam_np = cam[0].cpu().numpy()
    show_cam_overlay(input_np, cam_np, f'Grad-CAM: class {args.target_class}')
    # Save overlay as image
    plt.imsave(os.path.join(args.output_dir, "gradcam_overlay.png"), cam_np[:, cam_np.shape[1]//2, :], cmap="jet")
    # Prediction overlay
    with torch.no_grad():
        predictions = model(image)
    create_blended_visualizations(image, predictions, args.output_dir)
    print("Visualization generation complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()