# SwinUNETR-V2 Pipeline Architecture

## 1. Data Augmentation (`src/data/augmentations.py`)
- **Purpose:** Enhance model robustness and generalization.
- **Training Augmentations:**
  - Loading, channel formatting, orientation, spacing
  - Random spatial cropping, flipping, rotation
  - Intensity normalization, scaling, shifting
  - Optional: Gaussian noise, contrast adjustment
- **Validation Augmentations:**
  - Only essential preprocessing (no random augmentations)

## 2. Dataloader (`src/data/dataloader.py`)
- **Purpose:** Prepare PyTorch DataLoader objects for training and validation.
- **Key Steps:**
  - Loads a JSON datalist and splits it into train/val using `train_test_split` (random, reproducible split)
  - Applies the appropriate transforms
  - Returns DataLoader objects with correct shuffling and batching

## 3. Label Conversion (`src/data/convert_labels.py`)
- **Purpose:** Convert raw segmentation labels to multi-channel format (TC, WT, ET) for BraTS 2021/2023.
- **Logic:**
  - Maps dataset-specific label values to three binary channels

## 4. Model: SwinUNETR + Optional Modality Attention (`src/models/pipeline.py`)
- **Purpose:** 3D medical image segmentation using a transformer-based architecture.
- **Components:**
  - **ModalityAttentionModule (optional):** Learns channel/spatial attention across MRI modalities. Enabled via `--use_modality_attention`.
  - **SwinUNETR:** Main segmentation backbone. Configurable via CLI (feature size, depths, heads, etc).
- **Loss:**
  - Dice, DiceCE+Focal (hybrid), or Dice only (CLI toggle)
  - Optional class weights (CLI toggle)
- **Metrics:**
  - Dice, IoU, Hausdorff, precision, recall, F1
  - Per-class and mean metrics
- **Wandb Logging:**
  - Logs metrics and sample segmentation images (input, prediction, label) per validation epoch

## 5. Training Logic (`src/models/trainer.py`)
- **Purpose:** Orchestrate model training and validation using PyTorch Lightning.
- **Key Features:**
  - Early stopping (monitors val_mean_dice, configurable patience/min_delta)
  - Model checkpointing (local only, not wandb)
  - WandbLogger for experiment tracking
  - Passes all CLI/config arguments to the model

## 6. Main Entrypoint (`main.py`)
- **Purpose:** Defines the `main(args)` function, which sets up data, model, and trainer, and starts training.
- **Note:** No CLI parsing here; all configuration is passed from `kaggle_run.py`.

## 7. Experiment Runner (`kaggle_run.py`)
- **Purpose:** User-facing script for running experiments with full CLI support.
- **Features:**
  - Parses all relevant arguments (dataset, epochs, batch size, model/loss options, augmentation toggles, etc)
  - Prepares the environment and data
  - Constructs an `args` namespace and calls `main(args)`
  - Prints a summary of the configuration
  - Handles errors gracefully

## 8. Typical Flow
1. **User runs** `kaggle_run.py` with desired CLI arguments.
2. **Data is prepared** and split, augmentations are set up.
3. **Dataloaders** are created and passed to the model.
4. **Model** (SwinUNETR, optionally with Modality Attention) is initialized.
5. **Trainer** is set up with callbacks and wandb logging.
6. **Training/validation** proceeds, with metrics and images logged to wandb.
7. **Best model** (by Dice) is saved locally.

---

**For more details, see the respective files in `src/` and the CLI example in `kaggle_run.py`.** 