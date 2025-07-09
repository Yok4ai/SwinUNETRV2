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
        RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),


        # Affine transformations
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.4,
            shear_range=(0.5, 0.5, 0.5),
            padding_mode="zeros",
            # rotate_range=(0.1, 0.1, 0.1),  # Reduced rotation
            # scale_range=(0.05, 0.05, 0.05),  # Reduced scaling
            # translate_range=(5, 5, 5),  # Small translations
        ),

        # Intensity augmentations
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

        # Light Gaussian noise for robustness
        RandGaussianNoised(keys="image", prob=0.2, std=0.01),

        # Contrast adjustment - helps with tumor boundary detection         
        RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.8, 1.3)),  # Conservative range
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