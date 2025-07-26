# Standalone validation script with TTA
"""
Standalone validation script with Test Time Augmentation for SwinUNETR V2.
This script can be run independently, similar to kaggle_run.py.

Usage examples:

# Basic validation with TTA
python validate_with_tta.py \
    --checkpoint_path checkpoints/model-epoch-50.ckpt \
    --data_dir /path/to/brats_data

# Validation without TTA
python validate_with_tta.py \
    --checkpoint_path checkpoints/model.ckpt \
    --data_dir /path/to/data \
    --use_tta

# Custom parameters
python validate_with_tta.py \
    --checkpoint_path model.pth \
    --data_dir /path/to/data \
    --dataset brats2021 \
    --use_tta \
    --tta_merge_mode median \
    --roi_size 128 128 128 \
    --overlap 0.8 \
    --threshold 0.6 \
    --save_predictions \
    --log_to_wandb

# Fast validation (no TTA)
python validate_with_tta.py \
    --checkpoint_path model.ckpt \
    --data_dir /path/to/data \
    --batch_size 2 \
    --num_workers 8

# GPU selection
CUDA_VISIBLE_DEVICES=1 python validate_with_tta.py \
    --checkpoint_path model.ckpt \
    --data_dir /path/to/data \
    --device cuda:1
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the validation pipeline
from src.utils.tta import main

if __name__ == "__main__":
    # Print header
    print("="*80)
    print("SWINUNETR V2 STANDALONE VALIDATION WITH TEST TIME AUGMENTATION")
    print("="*80)
    print()
    
    # Run the validation
    results = main()
    
    print("\nValidation completed successfully!")
    print("="*80)