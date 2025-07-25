# SwinUNETR++ Usage Guide

This comprehensive guide explains how to use SwinUNETR++ with detailed parameter explanations and practical examples.

## Table of Contents
- [Quick Start](#quick-start)
- [CLI Arguments Overview](#cli-arguments-overview)
- [Parameter Categories](#parameter-categories)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Usage
```bash
# Simple training with default settings
python kaggle_run.py

# With custom dataset and epochs
python kaggle_run.py --dataset brats2023 --epochs 50 --batch_size 2
```

### Common Scenarios
```bash
# Beginner (fast, reliable)
python kaggle_run.py --loss_type dicece --use_class_weights --epochs 50

# Production (balanced performance)
python kaggle_run.py --loss_type generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5 --epochs 80

# Competition (maximum performance)
python kaggle_run.py --loss_type adaptive_progressive_hybrid --use_adaptive_scheduling \
  --structure_epochs 40 --boundary_epochs 70 --epochs 120 --use_class_weights
```

## CLI Arguments Overview

### How Arguments Work

SwinUNETR++ uses a two-stage configuration system:

1. **CLI Arguments**: Parsed using `argparse` and stored in `cli_args`
2. **Args Namespace**: Comprehensive configuration object passed to the training pipeline

```python
# CLI parsing
cli_args = parse_cli_args()

# Configuration namespace creation
args = argparse.Namespace(
    # CLI arguments are mapped here
    batch_size=cli_args.batch_size,
    loss_type=cli_args.loss_type,
    # Plus additional fixed parameters
    accelerator='gpu',
    precision='16-mixed',
    # ...
)
```

### Argument Types

| **Type** | **Example** | **Description** |
|:---------|:------------|:----------------|
| `--param value` | `--epochs 100` | Single value parameter |
| `--param val1 val2 val3` | `--roi_size 96 96 96` | Multiple values (nargs) |
| `--flag` | `--use_class_weights` | Boolean flag (store_true) |
| `--param choice` | `--loss_type dice` | Choice from predefined options |

## Parameter Categories

### 🗂️ Data Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--dataset` | choice | `brats2023` | Dataset version: `brats2021`, `brats2023` |
| `--batch_size` | int | `2` | Training batch size |
| `--num_workers` | int | `3` | Data loader workers |
| `--img_size` | int | `96` | Input image size (cubic) |
| `--roi_size` | int×3 | `96 96 96` | ROI size for sliding window inference |

**Usage Examples:**
```bash
# BraTS 2021 dataset with larger batch
python kaggle_run.py --dataset brats2021 --batch_size 4

# Larger images for better resolution
python kaggle_run.py --img_size 128 --roi_size 128 128 128
```

### 🧠 Model Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--feature_size` | int | `48` | Base feature dimension |
| `--use_modality_attention` | flag | `False` | Enable modality attention module |

**Usage Examples:**
```bash
# Larger model for better performance
python kaggle_run.py --feature_size 64

# Enable modality attention for multi-modal MRI
python kaggle_run.py --use_modality_attention
```

### ⚡ Training Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--epochs` | int | `100` | Total training epochs |
| `--learning_rate` | float | `1e-4` | Initial learning rate |
| `--warmup_epochs` | int | `10` | Learning rate warmup epochs |
| `--early_stopping_patience` | int | `15` | Early stopping patience |
| `--val_interval` | int | `1` | Validation frequency (epochs) |
| `--limit_val_batches` | int | `5` | Limit validation batches |

**Usage Examples:**
```bash
# Fast training with aggressive learning rate
python kaggle_run.py --epochs 50 --learning_rate 5e-4 --warmup_epochs 5

# Conservative training with patience
python kaggle_run.py --epochs 200 --learning_rate 5e-5 --early_stopping_patience 25
```

### 🎯 Loss Function Parameters

#### Core Loss Selection
| **Parameter** | **Type** | **Default** | **Options** |
|:--------------|:---------|:------------|:------------|
| `--loss_type` | choice | `dice` | `dice`, `dicece`, `dicefocal`, `generalized_dice`, `generalized_dice_focal`, `focal`, `tversky`, `hausdorff`, `hybrid_gdl_focal_tversky`, `hybrid_dice_hausdorff`, `adaptive_structure_boundary`, `adaptive_progressive_hybrid`, `adaptive_complexity_cascade`, `adaptive_dynamic_hybrid` |

#### Class Weighting
| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--use_class_weights` | flag | `False` | Enable class weighting |
| `--class_weights` | float×3 | `3.0 1.0 5.0` | Weights for TC, WT, ET classes |

#### Loss-Specific Parameters
| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--focal_gamma` | float | `2.0` | Focal loss focusing parameter |
| `--focal_alpha` | float | `None` | Focal loss class balancing |
| `--tversky_alpha` | float | `0.5` | Tversky FN weight |
| `--tversky_beta` | float | `0.5` | Tversky FP weight |
| `--hausdorff_alpha` | float | `2.0` | Hausdorff distance scaling |
| `--gdl_weight_type` | choice | `square` | GDL weighting: `square`, `simple`, `uniform` |

#### Loss Combination Weights
| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--lambda_dice` | float | `1.0` | Dice loss component weight |
| `--lambda_focal` | float | `1.0` | Focal loss component weight |
| `--lambda_tversky` | float | `1.0` | Tversky loss component weight |
| `--lambda_hausdorff` | float | `1.0` | Hausdorff loss component weight |
| `--gdl_lambda` | float | `1.0` | Generalized Dice loss weight |

**Usage Examples:**
```bash
# Focus on small lesions (ET)
python kaggle_run.py --loss_type generalized_dice_focal --use_class_weights \
  --class_weights 5.0 1.0 8.0 --focal_gamma 2.5

# Balanced hybrid approach
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky \
  --gdl_lambda 1.0 --lambda_focal 0.7 --lambda_tversky 0.5

# High sensitivity for tumor detection
python kaggle_run.py --loss_type tversky --tversky_alpha 0.3 --tversky_beta 0.7
```

### 📈 Adaptive Loss Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--use_adaptive_scheduling` | flag | `False` | Enable adaptive loss scheduling |
| `--adaptive_schedule_type` | choice | `linear` | Schedule type: `linear`, `exponential`, `cosine` |
| `--structure_epochs` | int | `30` | Structure learning phase duration |
| `--boundary_epochs` | int | `50` | Boundary refinement phase start |
| `--schedule_start_epoch` | int | `10` | When to start adaptive scheduling |
| `--min_loss_weight` | float | `0.1` | Minimum component weight |
| `--max_loss_weight` | float | `2.0` | Maximum component weight |

**Usage Examples:**
```bash
# Progressive learning with cosine scheduling
python kaggle_run.py --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling --adaptive_schedule_type cosine \
  --structure_epochs 40 --boundary_epochs 70

# Early adaptation with wide weight range
python kaggle_run.py --loss_type adaptive_structure_boundary \
  --use_adaptive_scheduling --schedule_start_epoch 5 \
  --min_loss_weight 0.05 --max_loss_weight 3.0
```

### 🔥 Local Minima Escape Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--use_aggressive_restart` | flag | `False` | Enable aggressive restart every epoch |
| `--escape_lr_multiplier` | float | `3.0` | LR boost factor for aggressive restart |
| `--use_warm_restarts` | flag | `False` | Enable standard warm restarts |
| `--restart_period` | int | `20` | Restart period in epochs |
| `--restart_mult` | int | `1` | Period multiplier after restart |

**Usage Examples:**
```bash
# Aggressive restart for breaking local minima (RECOMMENDED for stuck training)
python kaggle_run.py --use_aggressive_restart --escape_lr_multiplier 4.0

# Resume from checkpoint with aggressive restart
python kaggle_run.py --use_aggressive_restart --escape_lr_multiplier 3.0 \
  --learning_rate 5e-5 --resume_from_checkpoint checkpoints/model-epoch-036.ckpt

# Standard warm restarts with frequent periods
python kaggle_run.py --use_warm_restarts --restart_period 15

# Progressive restart periods
python kaggle_run.py --use_warm_restarts --restart_period 25 --restart_mult 2
```

#### 🚨 Breaking Local Minima from Checkpoints
```bash
# If your training gets stuck at 94% dice around epoch 36
python kaggle_run.py \
  --use_aggressive_restart \
  --escape_lr_multiplier 4.0 \
  --learning_rate 5e-5 \
  --epochs 60 \
  --resume_from_checkpoint checkpoints/your-checkpoint.ckpt
```

### 🔧 Inference Parameters

| **Parameter** | **Type** | **Default** | **Description** |
|:--------------|:---------|:------------|:----------------|
| `--overlap` | float | `0.7` | Sliding window overlap |
| `--threshold` | float | `0.5` | Post-processing threshold |

**Usage Examples:**
```bash
# High overlap for better boundary quality
python kaggle_run.py --overlap 0.8

# Lower threshold for higher sensitivity
python kaggle_run.py --threshold 0.4
```

## Usage Examples

### 🎯 Scenario-Based Examples

#### Beginner: Fast and Reliable
```bash
python kaggle_run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 2 \
  --loss_type dicece \
  --learning_rate 5e-4 \
  --use_class_weights
```

#### Standard Production: Balanced Performance
```bash
python kaggle_run.py \
  --dataset brats2023 \
  --epochs 80 \
  --batch_size 2 \
  --loss_type generalized_dice_focal \
  --learning_rate 1e-4 \
  --gdl_lambda 1.0 \
  --lambda_focal 0.5 \
  --focal_gamma 2.0 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0
```

#### Competition: Maximum Performance
```bash
python kaggle_run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --loss_type adaptive_progressive_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --adaptive_schedule_type cosine \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --schedule_start_epoch 10 \
  --use_warm_restarts \
  --restart_period 30 \
  --use_class_weights \
  --class_weights 5.0 1.0 8.0 \
  --use_modality_attention
```

### 🎨 Loss Function Specific Examples

#### Dice Loss (Simple Baseline)
```bash
python kaggle_run.py \
  --loss_type dice \
  --use_class_weights \
  --class_weights 3.0 1.0 5.0 \
  --epochs 50
```

#### DiceFocal (Class Imbalance)
```bash
python kaggle_run.py \
  --loss_type dicefocal \
  --lambda_dice 1.0 \
  --lambda_focal 1.0 \
  --focal_gamma 2.0 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0
```

#### Tversky (Precision/Recall Balance)
```bash
# High sensitivity (catch all tumors)
python kaggle_run.py \
  --loss_type tversky \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --use_class_weights

# High precision (clean boundaries)
python kaggle_run.py \
  --loss_type tversky \
  --tversky_alpha 0.7 \
  --tversky_beta 0.3 \
  --use_class_weights
```

#### Hybrid Combinations
```bash
# GDL + Focal + Tversky (Ultimate Hybrid)
python kaggle_run.py \
  --loss_type hybrid_gdl_focal_tversky \
  --gdl_lambda 1.0 \
  --lambda_focal 0.5 \
  --lambda_tversky 0.3 \
  --focal_gamma 2.0 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --use_class_weights

# Dice + Hausdorff (Volume + Boundary)
python kaggle_run.py \
  --loss_type hybrid_dice_hausdorff \
  --lambda_dice 1.0 \
  --lambda_hausdorff 0.1 \
  --hausdorff_alpha 2.0 \
  --use_class_weights
```

### 🚀 Advanced Adaptive Examples

#### Adaptive Structure-Boundary
```bash
python kaggle_run.py \
  --loss_type adaptive_structure_boundary \
  --use_adaptive_scheduling \
  --adaptive_schedule_type cosine \
  --schedule_start_epoch 15 \
  --min_loss_weight 0.2 \
  --max_loss_weight 1.5 \
  --epochs 100
```

#### Adaptive Progressive Hybrid
```bash
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --schedule_start_epoch 10 \
  --epochs 120
```

#### Adaptive Dynamic (Performance-Based)
```bash
python kaggle_run.py \
  --loss_type adaptive_dynamic_hybrid \
  --use_adaptive_scheduling \
  --schedule_start_epoch 20 \
  --epochs 100
```

### 🛠️ Hardware-Specific Examples

#### Memory-Constrained (Small GPU)
```bash
python kaggle_run.py \
  --batch_size 1 \
  --img_size 96 \
  --roi_size 96 96 96 \
  --limit_val_batches 3 \
  --epochs 75 \
  --loss_type dicece
```

#### High-Memory (Large GPU)
```bash
python kaggle_run.py \
  --batch_size 4 \
  --img_size 128 \
  --roi_size 128 128 128 \
  --feature_size 64 \
  --limit_val_batches 10 \
  --use_modality_attention
```

## Best Practices

### 🎯 Loss Function Selection

1. **Start Simple**: Begin with `dicece` for baseline
2. **Address Imbalance**: Use `generalized_dice_focal` for small lesions
3. **Fine-tune**: Apply hybrid losses for specific requirements
4. **Optimize**: Use adaptive losses for maximum performance

### ⚙️ Parameter Tuning Order

1. **Learning Rate & Batch Size**: Find stable training configuration
2. **Class Weights**: Based on dataset class distribution
3. **Loss-Specific Parameters**: gamma, alpha, beta values
4. **Loss Combination Weights**: lambda parameters
5. **Adaptive Scheduling**: Enable for final optimization

### 📊 Training Strategy

1. **Warmup**: Always use 10-15 warmup epochs
2. **Early Stopping**: Monitor with patience=15 for validation Dice
3. **Validation**: Use appropriate val_interval and limit_val_batches
4. **Checkpointing**: Enable for long training runs

### 🔧 Hardware Optimization

```bash
# For 16GB GPU
--batch_size 2 --img_size 96 --limit_val_batches 5

# For 24GB GPU  
--batch_size 4 --img_size 128 --limit_val_batches 8

# For 40GB+ GPU
--batch_size 8 --img_size 128 --feature_size 64 --use_modality_attention
```

## Troubleshooting

### 🚨 Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 1

# Reduce image size
--img_size 96 --roi_size 96 96 96

# Reduce validation batches
--limit_val_batches 3

# Disable modality attention
# (remove --use_modality_attention flag)
```

#### Loss Not Decreasing
```bash
# Increase learning rate
--learning_rate 5e-4

# Check class weights
--use_class_weights --class_weights 4.0 1.0 6.0

# Reduce complexity
--loss_type dicece  # Start with simpler loss
```

#### Poor Boundary Quality
```bash
# Use focal loss
--loss_type dicefocal --focal_gamma 2.5

# Add Hausdorff component
--loss_type hybrid_dice_hausdorff --lambda_hausdorff 0.1

# Increase overlap
--overlap 0.8
```

#### Missing Small Lesions
```bash
# Aggressive ET weighting
--class_weights 5.0 1.0 10.0

# Use Tversky for sensitivity
--loss_type tversky --tversky_alpha 0.3 --tversky_beta 0.7

# Generalized Dice
--loss_type generalized_dice_focal
```

### 🔍 Debug Commands

#### Check Configuration
```bash
# The script prints detailed configuration before training
python kaggle_run.py --help  # See all available parameters
```

#### Monitor Training
- Use WandB logging (automatic)
- Check validation metrics every epoch
- Monitor early stopping patience

#### Validate Results
- Check dice scores for all tumor regions (TC, WT, ET)
- Examine Hausdorff distances for boundary quality
- Review training curves for overfitting

## Configuration File Alternative

For complex configurations, consider creating a wrapper script:

```bash
#!/bin/bash
# sota_config.sh - State-of-the-art configuration

python kaggle_run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 2 \
  --img_size 96 \
  --feature_size 48 \
  --loss_type adaptive_progressive_hybrid \
  --learning_rate 1e-4 \
  --warmup_epochs 10 \
  --use_adaptive_scheduling \
  --adaptive_schedule_type cosine \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --schedule_start_epoch 10 \
  --min_loss_weight 0.1 \
  --max_loss_weight 2.0 \
  --use_warm_restarts \
  --restart_period 30 \
  --restart_mult 1 \
  --use_class_weights \
  --class_weights 5.0 1.0 8.0 \
  --use_modality_attention \
  --focal_gamma 2.5 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --early_stopping_patience 15 \
  --overlap 0.7 \
  --threshold 0.5
```

This comprehensive guide covers all aspects of using SwinUNETR++ effectively. For more details on specific loss functions, see [docs/loss.md](loss.md).