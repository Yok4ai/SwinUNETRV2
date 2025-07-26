# Standalone validation script with TTA
"""
Standalone validation script with Test Time Augmentation for SwinUNETR V2.
This script can be run independently for comprehensive model validation.

Usage examples:

# 1. Basic TTA validation with existing dataset.json
python validate_with_tta.py \
    --checkpoint_path /kaggle/input/model/pytorch/default/1/model.ckpt \
    --data_dir /kaggle/input/brats2023json/dataset.json \
    --dataset brats2023 \
    --use_tta

# 2. Domain shift testing: Train on 2023, test on 2021 with auto JSON creation
python validate_with_tta.py \
    --checkpoint_path /kaggle/input/swinv2-brats2023/model.ckpt \
    --input_dir /kaggle/input/brats2021-data \
    --dataset brats2021 \
    --use_tta \
    --max_batches 20

# 3. Fast validation (limited batches, no TTA)
python validate_with_tta.py \
    --checkpoint_path model.ckpt \
    --data_dir dataset.json \
    --dataset brats2023 \
    --max_batches 10 \
    --batch_size 2
    # Note: TTA is OFF by default

# 4. Full TTA validation with custom parameters
python validate_with_tta.py \
    --checkpoint_path /kaggle/input/model/model.ckpt \
    --data_dir /kaggle/input/data/dataset.json \
    --dataset brats2023 \
    --use_tta \
    --tta_merge_mode mean \
    --roi_size 96 96 96 \
    --overlap 0.7 \
    --threshold 0.5 \
    --save_predictions \
    --log_to_wandb

# 5. Cross-domain validation with WandB logging
python validate_with_tta.py \
    --checkpoint_path /kaggle/input/swinunetr-2023/model.ckpt \
    --input_dir /kaggle/input/brats2021-training \
    --dataset brats2021 \
    --use_tta \
    --max_batches 50 \
    --log_to_wandb \
    --wandb_project cross-domain-validation

# 6. Quick test (few batches, specific GPU)
CUDA_VISIBLE_DEVICES=0 python validate_with_tta.py \
    --checkpoint_path model.ckpt \
    --data_dir dataset.json \
    --dataset brats2023 \
    --max_batches 5 \
    --device cuda:0

Key Parameters:
- --checkpoint_path: Path to .ckpt or .pth model file
- --data_dir: Path to existing dataset.json OR directory containing it
- --input_dir: Path to raw BraTS data (creates dataset.json automatically)
- --dataset: brats2021 or brats2023 (for label format)
- --use_tta: Enable Test Time Augmentation (8 transforms, OFF by default)
- --max_batches: Limit validation batches for quick testing
- --tta_merge_mode: mean/median/max for TTA ensemble
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