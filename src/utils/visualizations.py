import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from monai.visualize import GradCAM
from monai.utils.misc import set_determinism
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Project-specific imports
from models.swinunetr import SwinUNETR
from data.augmentations import get_transforms
from data.dataloader import get_dataloaders
from kaggle_setup import prepare_brats_data
from captum.attr import Saliency
from captum.attr import IntegratedGradients

set_determinism(42)

def load_datalist(path, max_samples=100):
    with open(path) as f:
        return json.load(f)["training"][:max_samples]

def build_model(img_size, in_channels, out_channels, feature_size, use_v2=True):
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=2,
        feature_size=feature_size,
        use_checkpoint=True,
        use_v2=use_v2,
        spatial_dims=3,
    )

def load_weights(model, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
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

def show_cam_overlay(image, cam, title, channel_idx=3, save_path=None):
    channel_names = ["T1c", "T1n", "T2f", "T2w"]
    channel_name = channel_names[channel_idx] if 0 <= channel_idx < len(channel_names) else f"Channel {channel_idx}"
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
        print(f"[INFO] Saved Grad-CAM overlay to {save_path}")
    else:
        plt.show()
    plt.close()

def run_gradcam(
    dataset_path,
    checkpoint_path,
    sample_idx=0,
    target_class=1,
    target_layer="encoder1",
    output_dir=".",
    channel_idx=3
):
    # Load datalist
    datalist = load_datalist(dataset_path)
    # Use project-specific transforms, always with img_size=96 for compatibility
    train_tfms, val_tfms = get_transforms(img_size=96)
    from monai.data import Dataset, DataLoader
    dataset = Dataset(data=datalist, transform=val_tfms)
    loader = DataLoader(dataset, batch_size=1)

    # Model
    model = build_model(img_size=96, in_channels=4, out_channels=3, feature_size=48, use_v2=True)
    model, device = load_weights(model, checkpoint_path)

    # Get sample
    sample = dataset[sample_idx]
    image = sample["image"].unsqueeze(0).to(device)

    # Force resize to [96, 96, 96] if needed
    if image.shape[2:] != (96, 96, 96):
        image = F.interpolate(image, size=(96, 96, 96), mode="trilinear", align_corners=False)

    # GradCAM
    gradcam = GradCAM(nn_module=model, target_layers=target_layer)
    cam_raw = gradcam(x=image, class_idx=target_class)

    # Resize and Normalize CAM
    cam = resize_cam(cam_raw, image.shape[2:])

    # Visualization
    input_np = image[0].cpu().numpy()
    cam_np = cam[0].cpu().numpy()
    class_names = ["Tumor Core", "Whole Tumor", "Enhancing Tumor"]
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"gradcam_sample{sample_idx}_class{target_class}_layer{target_layer}_channel{channel_idx}.png")
    show_cam_overlay(input_np, cam_np, f"Grad-CAM: {class_names[target_class]}", channel_idx=channel_idx, save_path=save_path)
    return input_np, cam_np

def run_saliency(
    dataset_path,
    checkpoint_path,
    sample_idx=0,
    target_class=1,
    output_dir=".",
    channel_idx=3
):
    datalist = load_datalist(dataset_path)
    train_tfms, val_tfms = get_transforms(img_size=96)
    from monai.data import Dataset
    dataset = Dataset(data=datalist, transform=val_tfms)
    model = build_model(img_size=96, in_channels=4, out_channels=3, feature_size=48, use_v2=True)
    model, device = load_weights(model, checkpoint_path)
    model.eval()
    sample = dataset[sample_idx]
    image = sample["image"].unsqueeze(0).to(device)
    image.requires_grad = True
    # Force resize to [96, 96, 96] if needed
    if image.shape[2:] != (96, 96, 96):
        image = F.interpolate(image, size=(96, 96, 96), mode="trilinear", align_corners=False)
    def forward_func(x):
        output = model(x)
        # Aggregate over all voxels for the target class and keep batch dimension
        return output[:, target_class].sum().unsqueeze(0)
    saliency = Saliency(forward_func)
    attributions = saliency.attribute(image, target=None)
    attributions = attributions.detach().cpu().numpy()[0]
    input_np = image[0].detach().cpu().numpy()
    mid = input_np.shape[2] // 2
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_np[channel_idx, :, mid, :], cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(input_np[channel_idx, :, mid, :], cmap="gray")
    attr_norm = normalize_attr(attributions[channel_idx, :, mid, :])
    plt.imshow(attr_norm, cmap="jet", alpha=0.5)
    plt.title("Saliency Overlay")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"saliency_sample{sample_idx}_class{target_class}_channel{channel_idx}.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved saliency overlay to {save_path}")
    plt.close()
    return input_np, attributions

def normalize_attr(attr):
    attr = np.abs(attr)
    attr = (attr - np.min(attr)) / (np.max(attr) - np.min(attr) + 1e-8)
    return attr

def run_integrated_gradients(
    dataset_path,
    checkpoint_path,
    sample_idx=0,
    target_class=1,
    output_dir=".",
    channel_idx=3,
    cmap="magma"
):
    datalist = load_datalist(dataset_path)
    train_tfms, val_tfms = get_transforms(img_size=96)
    from monai.data import Dataset
    dataset = Dataset(data=datalist, transform=val_tfms)
    model = build_model(img_size=96, in_channels=4, out_channels=3, feature_size=48, use_v2=True)
    model, device = load_weights(model, checkpoint_path)
    model.eval()
    sample = dataset[sample_idx]
    image = sample["image"].unsqueeze(0).to(device)
    image.requires_grad = True
    # Force resize to [96, 96, 96] if needed
    if image.shape[2:] != (96, 96, 96):
        image = F.interpolate(image, size=(96, 96, 96), mode="trilinear", align_corners=False)
    def forward_func(x):
        output = model(x)
        return output[:, target_class].sum().unsqueeze(0)
    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(image, target=None, n_steps=4)
    attributions = attributions.detach().cpu().numpy()[0]
    input_np = image[0].detach().cpu().numpy()
    mid = input_np.shape[2] // 2
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_np[channel_idx, :, mid, :], cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(input_np[channel_idx, :, mid, :], cmap="gray")
    attr_norm = normalize_attr(attributions[channel_idx, :, mid, :])
    plt.imshow(attr_norm, cmap=cmap, alpha=0.8)
    plt.title("Integrated Gradients Overlay")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"integrated_gradients_sample{sample_idx}_class{target_class}_channel{channel_idx}.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved Integrated Gradients overlay to {save_path}")
    plt.close()
    return input_np, attributions


def main():
    parser = argparse.ArgumentParser(description="SwinUNETR V2 Visualization (GradCAM/Saliency)")
    parser.add_argument("--dataset_path", type=str, help="Path to dataset.json (will be created if --prepare_json is set)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--targetclass", type=int, default=1, help="Target class index (0=TC, 1=WT, 2=ET)")
    parser.add_argument("--target_layer", type=str, default="encoder1", help="Target layer for GradCAM (e.g., encoder1, encoder2, etc.)")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--prepare_json", action="store_true", help="If set, create dataset.json using prepare_brats_data before running visualization.")
    parser.add_argument("--input_dir", type=str, help="Input directory with subject folders (for --prepare_json)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/visualizations", help="Output directory for dataset.json (for --prepare_json) and outputs")
    parser.add_argument("--dataset_type", type=str, default="brats2023", choices=["brats2021", "brats2023"], help="Dataset type (for --prepare_json)")
    parser.add_argument("--channel_idx", type=int, default=3, help="Image channel index to visualize (0=T1c, 1=T1n, 2=T2f, 3=T2w)")
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "saliency", "integrated_gradients"], help="Visualization method: gradcam, saliency, or integrated_gradients")
    parser.add_argument("--cmap", type=str, default="magma", help="Colormap for attribution overlay (e.g., magma, jet, hot, viridis)")
    args = parser.parse_args()

    dataset_json_path = args.dataset_path
    if args.prepare_json:
        if not args.input_dir:
            raise ValueError("--input_dir must be specified when using --prepare_json.")
        prepare_brats_data(args.input_dir, args.output_dir, args.dataset_type)
        dataset_json_path = f"{args.output_dir}/dataset.json"
        print(f"[INFO] Created dataset.json at {dataset_json_path}")

    if not dataset_json_path:
        raise ValueError("--dataset_path must be specified (or use --prepare_json to create it).")

    if args.method == "gradcam":
        run_gradcam(
            dataset_path=dataset_json_path,
            checkpoint_path=args.checkpoint_path,
            sample_idx=args.sample_idx,
            target_class=args.targetclass,
            target_layer=args.target_layer,
            output_dir=args.output_dir,
            channel_idx=args.channel_idx
        )
    elif args.method == "saliency":
        run_saliency(
            dataset_path=dataset_json_path,
            checkpoint_path=args.checkpoint_path,
            sample_idx=args.sample_idx,
            target_class=args.targetclass,
            output_dir=args.output_dir,
            channel_idx=args.channel_idx
        )
    elif args.method == "integrated_gradients":
        run_integrated_gradients(
            dataset_path=dataset_json_path,
            checkpoint_path=args.checkpoint_path,
            sample_idx=args.sample_idx,
            target_class=args.targetclass,
            output_dir=args.output_dir,
            channel_idx=args.channel_idx,
            cmap=args.cmap
        )
    else:
        raise ValueError(f"Unknown visualization method: {args.method}")

if __name__ == "__main__":
    main()
