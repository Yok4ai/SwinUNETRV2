from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandRotate90d,
    RandAffined,
)
from data.convert_labels import ConvertLabels

def get_transforms(img_size, dataset="brats2023"):
    """Get training and validation transforms for BraTS data."""
    train_transforms = Compose(
        [
        # Essential loading and preprocessing
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertLabels(keys="label", dataset=dataset),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # Spatial cropping
        RandSpatialCropd(keys=["image", "label"], roi_size=[img_size, img_size, img_size], random_size=False),

        # Geometric augmentations - proven to boost dice
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),

        # Intensity augmentations
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

        ## Light Gaussian noise for robustness
        # RandGaussianNoised(keys="image", prob=0.2, std=0.01),

        # Contrast adjustment - helps with tumor boundary detection         
        RandAdjustContrastd(keys="image", prob=0.1, gamma=(0.8, 1.3)),  # Conservative range
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertLabels(keys="label", dataset=dataset),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    return train_transforms, val_transforms

def get_tta_transforms():
    """Get Test Time Augmentation transforms for brain tumor segmentation."""
    tta_transforms = Compose([
        # Spatial flips - most reliable for brain images
        RandFlipd(keys="image", prob=0.5, spatial_axis=0),
        RandFlipd(keys="image", prob=0.5, spatial_axis=1), 
        RandFlipd(keys="image", prob=0.5, spatial_axis=2),
        # Light rotations - conservative for medical images
        RandRotate90d(keys="image", prob=0.3, spatial_axes=(0, 1)),
        # Very light affine transformations
        RandAffined(
            keys="image",
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.05, 0.05, 0.05),
            translate_range=(2, 2, 2),
            padding_mode="border",
        ),
        # Minimal intensity augmentations for TTA
        RandScaleIntensityd(keys="image", factors=0.05, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.05, prob=0.3),
    ])
    
    return tta_transforms