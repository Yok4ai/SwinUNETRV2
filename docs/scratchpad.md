# SwinUNETR Pipeline Scratchpad

## Quick Reference

- **Augmentations:**
  - Training: random crop, flip, rotate, intensity, noise, contrast
  - Validation: only normalization
- **Dataloader:**
  - Uses sklearn's train_test_split for reproducible 80/20 split
- **Label Conversion:**
  - Converts raw labels to 3-channel (TC, WT, ET)
- **Model:**
  - SwinUNETR backbone
  - Optional Modality Attention (toggle: --use_modality_attention)
- **Loss:**
  - Dice only or hybrid (DiceCE+Focal), toggle: --loss_type
  - Optional class weights: --use_class_weights
- **Logging:**
  - wandb logs metrics and sample images (input, pred, label)
- **Entrypoint:**
  - Run everything via `kaggle_run.py` with CLI args

## Example CLI

See the example block in `kaggle_run.py` for a full command. 