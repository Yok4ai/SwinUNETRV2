# Project Architecture

## Folder Structure
- `src/data/`
  - `augmentations.py`: Defines MONAI-based data augmentations for training and validation.
  - `dataloader.py`: Loads data from BraTS-format `dataset.json`, splits into train/val, applies transforms.
  - `convert_labels.py`: Converts BraTS labels to multi-channel format (TC, WT, ET).
- `src/models/`
  - `swinunetr.py`: Implements the SwinUNETR-V2 model (3D Swin Transformer + UNet blocks).
  - `pipeline.py`: PyTorch Lightning module for training, validation, and metric logging.
  - `trainer.py`: Sets up data loaders, callbacks, and Lightning Trainer.
- `src/utils/`
  - `visualization.py`: Visualization utilities for predictions and training curves.
- `main.py`: Entry point for training, argument parsing, and pipeline orchestration.

## Data Flow
1. **Transforms**: `get_transforms` in `augmentations.py` defines training/validation pipelines (spatial, intensity, geometric, and label conversion).
2. **Data Loading**: `dataloader.py` loads and splits data, applies transforms, returns MONAI Datasets.
3. **Label Conversion**: `convert_labels.py` maps BraTS labels to 3 output channels (TC, WT, ET).

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
- Change model config in `main.py` or via CLI args.
- Add new metrics/losses in `pipeline.py`.
- Update docs for all major changes. 