# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SwinUNETR V2 is a PyTorch implementation for medical image segmentation using MONAI and PyTorch Lightning. The project focuses on brain tumor segmentation with support for BRATS 2021 and 2023 datasets.

## Key Commands

### Installation
```bash
# Local development
pip install -e .

# Kaggle environment
pip install -q ./SwinUNETRV2
```

### Training
```bash
# Main training command
python kaggle_run.py

# With custom parameters
python kaggle_run.py --dataset brats2023 --epochs 100 --batch_size 2 --learning_rate 1e-4 --loss_type dice --use_class_weights --use_modality_attention

# Direct module usage (advanced)
python -m swinunetrv2.main --input_dir /path/to/data --batch_size 8 --epochs 100
```

### Testing
No formal test suite exists. Model validation occurs during training via PyTorch Lightning validation loops.

## Architecture Overview

### Core Pipeline Flow
1. **kaggle_run.py**: Entry point that sets up environment and parameters, then calls main()
2. **kaggle_setup.py**: Prepares BraTS dataset and creates dataset.json for both 2021/2023 formats
3. **main.py**: Orchestrates the training pipeline:
   - Calls `get_transforms()` from `src/data/augmentations.py` 
   - Calls `get_dataloaders()` from `src/data/dataloader.py`
   - Calls `setup_training()` and `train_model()` from `src/models/trainer.py`
4. **src/models/trainer.py**: Sets up PyTorch Lightning trainer with callbacks and logging
5. **src/models/pipeline.py**: Contains `BrainTumorSegmentation` wrapper around SwinUNETR model
6. **src/models/swinunetr.py**: Core SwinUNETR V2 model implementation (note: uses MONAI's SwinUNETR)

### Key Modules
- **src/data/**: Data loading, augmentation, and label conversion with dataset-specific support
  - `dataloader.py`: Creates train/val DataLoaders with 80/20 split from dataset.json
  - `augmentations.py`: MONAI transforms for training (spatial/intensity augmentations) and validation
  - `convert_labels.py`: Converts between BRATS 2021/2023 label formats
- **src/models/**: SwinUNETR architecture, PyTorch Lightning pipeline, and training setup
  - `pipeline.py`: `BrainTumorSegmentation` Lightning module with metrics, loss functions, and training logic
  - `trainer.py`: Sets up trainer with early stopping, checkpointing, and WandB logging
- **src/utils/**: Visualization utilities for predictions and training curves

### Configuration System
All parameters are defined in `kaggle_run.py` as an `argparse.Namespace` object and passed to `main()`. CLI arguments override default values for key parameters like dataset, epochs, batch_size, learning_rate, etc. The system supports both Kaggle-specific paths and local development.

## Dataset Support
- Supports BRATS 2021 and 2023 label conventions via `--dataset` parameter
- Data expected in BraTS format with `dataset.json` file
- 80/20 train/validation split handled automatically

## Model Features
- SwinUNETR V2 architecture with configurable depths, heads, and feature sizes
- Hybrid loss: DiceCE + Focal loss with optional class weighting
- Optional modality attention module for better cross-modal feature extraction
- PyTorch Lightning integration with early stopping, checkpointing, and wandb logging
- Comprehensive metrics: Dice, IoU, Hausdorff distance, precision, recall, F1
- Sliding window inference for validation with configurable overlap

## Development Workflow

### Code Style (from docs/cursor.md)
- Use PEP8 for Python code with docstrings and type hints
- Use snake_case for functions/variables, PascalCase for classes
- Avoid hardcoding values; use arguments/configs
- Document all public functions and classes

### Key Implementation Details
- **BrainTumorSegmentation** (src/models/pipeline.py:62): Main PyTorch Lightning module
- **ModalityAttentionModule** (src/models/pipeline.py:24): Optional attention for MRI modalities
- **ConvertLabels** (src/data/convert_labels.py): Handles BRATS 2021/2023 label differences
- Model uses MONAI's SwinUNETR, not a custom implementation
- Training uses AdamW optimizer with warmup + cosine annealing scheduler
- Class weights: [1.0, 3.0, 5.0] for background, WT, TC, ET to handle imbalance

## Important Notes
- Use `kaggle_run.py` as the primary entry point for training
- All model parameters are configurable via the args namespace in `kaggle_run.py`
- The project uses MONAI for medical image processing transforms and utilities
- Training automatically uses mixed precision (16-bit) and DDP strategy when multiple GPUs available
- Checkpoints saved in `checkpoints/` directory with wandb logging to "brain-tumor-segmentation" project
- Early stopping monitors "val_mean_dice" with patience=15 epochs
- The pipeline supports both Kaggle (/kaggle/input/, /kaggle/working) and local paths