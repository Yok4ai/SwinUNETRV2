# Project Architecture

## Folder Structure
- `src/data/`
  - `augmentations.py`: Defines MONAI-based data augmentations for training and validation. Now supports a `dataset` argument to select label conventions (BRATS 2021 or 2023).
  - `dataloader.py`: Loads data from BraTS-format `dataset.json`, splits into train/val, applies transforms.
  - `convert_labels.py`: Converts BraTS labels to multi-channel format (TC, WT, ET). Supports both BRATS 2021 (ET=4) and BRATS 2023 (ET=3) via a `dataset` argument.
- `src/models/`
  - `swinunetr.py`: Implements the SwinUNETR-V2 model (3D Swin Transformer + UNet blocks).
  - `pipeline.py`: PyTorch Lightning module for training, validation, and metric logging.
  - `trainer.py`: Sets up data loaders, callbacks, and Lightning Trainer.
- `src/utils/`
  - `visualization.py`: Visualization utilities for predictions and training curves.
- `main.py`: Entry point for training, argument parsing, and pipeline orchestration. Passes the `dataset` argument to the data pipeline.

## Data Flow
1. **Transforms**: `get_transforms` in `augmentations.py` defines training/validation pipelines (spatial, intensity, geometric, and label conversion). The `dataset` argument is passed to select the correct label mapping for BRATS 2021 or 2023.
2. **Data Loading**: `dataloader.py` loads and splits data, applies transforms, returns MONAI Datasets.
3. **Label Conversion**: `convert_labels.py` maps BraTS labels to 3 output channels (TC, WT, ET), with the ET label set to 3 (BRATS 2023) or 4 (BRATS 2021) depending on the `dataset` argument.

## Model Architecture
- **SwinUNETR-V2** (`swinunetr.py`):
  - 3D Swin Transformer backbone with hierarchical attention.
  - UNet-style skip connections and upsampling blocks.
  - Configurable: input channels, output channels, feature size, depths, heads, window size, etc.
  - V2: Adds residual conv blocks at each Swin stage.

## Training Pipeline
- **Lightning Module** (`pipeline.py`):
  - Handles forward, loss (DiceCE + Focal), metrics (Dice, per-class), and logging.
  - Uses sliding window inference for validation.
  - Stores best model by validation Dice.
- **Trainer Setup** (`trainer.py`):
  - DataLoader setup with configurable batch size, workers, pin_memory, etc.
  - Early stopping, checkpointing, and wandb logging.
  - Supports AMP, DDP, and gradient clipping.

## Utilities
- **Visualization** (`visualization.py`):
  - Batch and prediction visualization.
  - Training/validation curve plotting.

## Extending/Modifying
- Add new transforms in `augmentations.py`.
- Change model config in `main.py` or via CLI args (including `dataset`).
- Add new metrics/losses in `pipeline.py`.
- Update docs for all major changes. 