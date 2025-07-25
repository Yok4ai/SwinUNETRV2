# Learning Rate Scheduling & Local Minima Escape

This guide covers advanced learning rate scheduling strategies and local minima escape techniques in SwinUNETR++.

## Table of Contents
- [Overview](#overview)
- [Standard Scheduling](#standard-scheduling)
- [Warm Restarts](#warm-restarts)
- [Aggressive Restart (Local Minima Escape)](#aggressive-restart-local-minima-escape)
- [Adaptive Loss Scheduling](#adaptive-loss-scheduling)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

SwinUNETR++ provides several learning rate scheduling strategies to optimize training:

1. **Standard Warmup + Cosine Annealing**: Smooth training progression
2. **Warm Restarts**: Periodic LR resets to escape local minima
3. **Aggressive Restart**: Every-epoch LR resets for extreme exploration
4. **Adaptive Loss Scheduling**: Dynamic loss component weighting

## Standard Scheduling

### Default Behavior
```python
# Standard warmup + cosine annealing (DEFAULT)
python kaggle_run.py --learning_rate 1e-4 --warmup_epochs 10 --epochs 100
```

**Schedule Pattern:**
- **Epochs 0-10**: Linear warmup from 0 → 1e-4
- **Epochs 10-100**: Cosine annealing 1e-4 → ~1e-6

### Configuration
```bash
# Custom warmup period
python kaggle_run.py --warmup_epochs 15 --learning_rate 2e-4

# No warmup (start immediately at full LR)
python kaggle_run.py --warmup_epochs 0 --learning_rate 1e-4
```

## Warm Restarts

### Theory
Warm restarts periodically reset the learning rate to its initial value, allowing the model to:
- Escape local minima
- Explore new regions of the loss landscape
- Maintain training momentum

### Basic Usage
```bash
# Enable warm restarts every 20 epochs
python kaggle_run.py --use_warm_restarts --restart_period 20

# More frequent restarts (every 10 epochs)
python kaggle_run.py --use_warm_restarts --restart_period 10

# Progressive restart periods (10 → 20 → 40 epochs)
python kaggle_run.py --use_warm_restarts --restart_period 10 --restart_mult 2
```

### Schedule Visualization
```
Epochs:    0    10   20   30   40   50   60
LR:        |    ^    |    ^    |    ^    |
           warmup   restart   restart   restart
```

### Advanced Configuration
```bash
# Conservative: Long periods with gradual extension
python kaggle_run.py --use_warm_restarts --restart_period 30 --restart_mult 1.5

# Aggressive: Frequent restarts with fixed periods  
python kaggle_run.py --use_warm_restarts --restart_period 8 --restart_mult 1

# Hybrid: Combine with adaptive scheduling
python kaggle_run.py --use_warm_restarts --restart_period 25 \
  --use_adaptive_scheduling --loss_type adaptive_progressive_hybrid
```

## Aggressive Restart (Local Minima Escape)

### When to Use
- **Training plateaued**: Validation dice stuck at ~94% for multiple epochs
- **Checkpoint resuming**: Need to break through previous local minima
- **Competition**: Seeking that extra 1-2% performance boost

### Theory
Aggressive restart resets learning rate **every single epoch**:
- Maximum exploration potential
- Prevents settling into any local minimum
- High volatility but can find better global solutions

### Basic Usage
```bash
# Enable aggressive restart (restarts every epoch)
python kaggle_run.py --use_aggressive_restart

# Custom LR multiplier (3x base LR on each restart)
python kaggle_run.py --use_aggressive_restart --escape_lr_multiplier 3.0

# Higher multiplier for stronger exploration (4x base LR)
python kaggle_run.py --use_aggressive_restart --escape_lr_multiplier 4.0
```

### Schedule Pattern
```
Epoch:  37    38    39    40    41    42
LR:     2e-4  2e-4  2e-4  2e-4  2e-4  2e-4  ← Restart to 2e-4 every epoch
        ↓     ↓     ↓     ↓     ↓     ↓
        5e-8  5e-8  5e-8  5e-8  5e-8  5e-8  ← Drops within epoch
```

### Checkpoint Resume Example
```bash
# Your model stuck at 94.32% dice at epoch 36
python kaggle_run.py \
  --use_aggressive_restart \
  --escape_lr_multiplier 4.0 \
  --learning_rate 5e-5 \
  --epochs 60 \
  --resume_from_checkpoint checkpoints/model-epoch-036.ckpt
```

**What happens:**
- **Base LR**: 5e-5 (conservative for pretrained model)
- **Restart LR**: 2e-4 (5e-5 × 4.0 = aggressive exploration)
- **Every epoch**: Jumps to 2e-4, drops to 5e-8, jumps again

### Parameter Guidelines

| **Scenario** | **escape_lr_multiplier** | **learning_rate** | **Notes** |
|:-------------|:-------------------------|:------------------|:----------|
| Fresh training | 2.0-3.0 | 1e-4 | Moderate exploration |
| Checkpoint resume | 3.0-4.0 | 5e-5 | Lower base, higher multiplier |
| Desperate escape | 4.0-6.0 | 3e-5 | Maximum exploration |
| Fine-tuning | 1.5-2.0 | 1e-5 | Gentle perturbation |

## Adaptive Loss Scheduling

### Overview
Adaptive scheduling dynamically adjusts **loss component weights** over time, not just learning rates.

### Available Types
1. **adaptive_structure_boundary**: Dice → Focal transition
2. **adaptive_progressive_hybrid**: Dice → Focal → Hausdorff progression  
3. **adaptive_complexity_cascade**: Multi-stage complexity introduction
4. **adaptive_dynamic_hybrid**: Performance-based adaptation

### Progressive Hybrid Example
```bash
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --structure_epochs 30 \
  --boundary_epochs 50 \
  --schedule_start_epoch 10
```

**Schedule:**
- **Epochs 0-10**: Standard dice loss (warmup)
- **Epochs 10-30**: Dice dominant (structure learning)  
- **Epochs 30-50**: Add focal loss (boundary refinement)
- **Epochs 50+**: Add Hausdorff loss (fine details)

### Combination with LR Scheduling
```bash
# Ultimate optimization: Adaptive loss + Aggressive restart
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --use_aggressive_restart \
  --escape_lr_multiplier 3.0 \
  --structure_epochs 40 \
  --boundary_epochs 70
```

## Best Practices

### 1. Standard Training (Stable & Reliable)
```bash
python kaggle_run.py \
  --learning_rate 1e-4 \
  --warmup_epochs 10 \
  --epochs 100
```

### 2. Local Minima Prevention
```bash
python kaggle_run.py \
  --use_warm_restarts \
  --restart_period 15 \
  --learning_rate 1e-4 \
  --epochs 120
```

### 3. Checkpoint Recovery (Stuck Training)
```bash
python kaggle_run.py \
  --use_aggressive_restart \
  --escape_lr_multiplier 3.5 \
  --learning_rate 5e-5 \
  --epochs 80 \
  --resume_from_checkpoint checkpoints/stuck-model.ckpt
```

### 4. Competition Optimization (Maximum Performance)
```bash
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --use_warm_restarts \
  --restart_period 25 \
  --learning_rate 1e-4 \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --epochs 150
```

## Troubleshooting

### 🚨 Common Issues

#### Training Stuck at Plateau
**Symptoms:** Validation dice stuck at same value for 10+ epochs

**Solutions:**
```bash
# Option 1: Aggressive restart
python kaggle_run.py --use_aggressive_restart --escape_lr_multiplier 4.0

# Option 2: Frequent warm restarts  
python kaggle_run.py --use_warm_restarts --restart_period 8

# Option 3: Higher base learning rate
python kaggle_run.py --learning_rate 2e-4 --use_warm_restarts
```

#### Aggressive Restart Too Volatile
**Symptoms:** Loss oscillating wildly, no convergence

**Solutions:**
```bash
# Reduce multiplier
python kaggle_run.py --use_aggressive_restart --escape_lr_multiplier 2.0

# Use standard warm restarts instead
python kaggle_run.py --use_warm_restarts --restart_period 5

# Lower base learning rate
python kaggle_run.py --use_aggressive_restart --learning_rate 3e-5
```

#### Poor Performance After Restart
**Symptoms:** Performance drops after LR restart

**Solutions:**
```bash
# Longer warmup period
python kaggle_run.py --warmup_epochs 15

# Gradual restart periods
python kaggle_run.py --use_warm_restarts --restart_period 20 --restart_mult 1.5

# Combined approach
python kaggle_run.py --use_warm_restarts --restart_period 30 \
  --use_adaptive_scheduling
```

### 🎯 Performance Optimization Tips

#### For Different Training Stages

**Early Training (0-30 epochs):**
```bash
# Focus on stability
--learning_rate 1e-4 --warmup_epochs 10
```

**Mid Training (30-70 epochs):**
```bash  
# Add exploration
--use_warm_restarts --restart_period 20
```

**Late Training (70+ epochs):**
```bash
# Break local minima
--use_aggressive_restart --escape_lr_multiplier 3.0
```

#### Hardware Considerations

**Low Memory (Limited batch size):**
```bash
# More frequent restarts compensate for small batches
--use_warm_restarts --restart_period 10
```

**High Memory (Large batch size):**
```bash
# Standard scheduling works well with stable gradients
--learning_rate 1e-4 --warmup_epochs 15
```

### 🔍 Monitoring & Debugging

#### Key Metrics to Watch
- **Learning Rate**: Should vary according to schedule
- **Validation Dice**: Should show improvement after restarts
- **Loss Trajectory**: Expect jumps at restart points
- **Training Stability**: Monitor for excessive oscillation

#### WandB Logging
All scheduling strategies automatically log:
- Current learning rate per epoch
- Loss component weights (adaptive scheduling)
- Restart events and their effects

#### Debug Commands
```bash
# Print detailed configuration
python kaggle_run.py --help

# Monitor learning rate changes
# (Check WandB dashboard or training logs)
```

## Advanced Combinations

### Multi-Stage Strategy
```bash
# Stage 1: Stable training
python kaggle_run.py --epochs 50 --learning_rate 1e-4

# Stage 2: Warm restart exploration  
python kaggle_run.py --epochs 100 --use_warm_restarts \
  --restart_period 15 --resume_from_checkpoint checkpoints/epoch-050.ckpt

# Stage 3: Aggressive final push
python kaggle_run.py --epochs 120 --use_aggressive_restart \
  --escape_lr_multiplier 3.0 --resume_from_checkpoint checkpoints/epoch-100.ckpt
```

### Competition Pipeline
```bash
#!/bin/bash
# competition_schedule.sh - Multi-phase optimization

# Phase 1: Structure learning (conservative)
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --learning_rate 1e-4 \
  --epochs 60 \
  --structure_epochs 30 \
  --boundary_epochs 45

# Phase 2: Exploration (warm restarts)  
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --use_warm_restarts \
  --restart_period 12 \
  --learning_rate 8e-5 \
  --epochs 100 \
  --resume_from_checkpoint checkpoints/epoch-060.ckpt

# Phase 3: Final optimization (aggressive)
python kaggle_run.py \
  --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --use_aggressive_restart \
  --escape_lr_multiplier 4.0 \
  --learning_rate 3e-5 \
  --epochs 120 \
  --resume_from_checkpoint checkpoints/epoch-100.ckpt
```

This comprehensive scheduling guide should help you optimize training performance and escape local minima effectively. For more details on loss functions, see [docs/loss.md](loss.md).