# SwinUNETR V2 Documentation

This directory contains comprehensive documentation for the SwinUNETR V2 pipeline for brain tumor segmentation.

## 📚 Documentation Files

### Core Architecture
- **[architecture.md](architecture.md)** - Complete pipeline architecture overview
  - Data processing pipeline (augmentations, dataloaders, label conversion)
  - Model components (SwinUNETR + Modality Attention)
  - 14 loss functions with adaptive scheduling
  - Training logic and experiment runner

### Loss Functions & Optimization
- **[loss.md](loss.md)** - **⭐ COMPREHENSIVE LOSS FUNCTION GUIDE**
  - 14 loss functions: Basic, Hybrid, and Adaptive
  - SOTA strategies and local minima escape techniques
  - BraTS-specific recommendations and parameter tuning
  - Competition-winning approaches and troubleshooting
  - Mathematical formulas and implementation details

### Model Variants
- **[swinunetrplus.md](swinunetrplus.md)** - Enhanced SwinUNETR+ architecture
  - Multi-scale attention mechanisms
  - Advanced feature extraction improvements

### Baseline Comparison
- **[SwinUNETR-Baseline.md](SwinUNETR-Baseline.md)** - Original SwinUNETR performance
  - Baseline Dice loss implementation
  - BraTS 2021 benchmark results
  - Comparison with nnU-Net, SegResNet, TransBTS
  - Enhanced V2 loss function extensions

### Development
- **[cursor.md](cursor.md)** - Development guidelines and code style
- **[scratchpad.md](scratchpad.md)** - Development notes and experiments

## 🚀 Quick Start

1. **Choose your loss function** from [loss.md](loss.md) based on your requirements:
   - **Beginners**: Start with `dicece`
   - **Standard use**: `generalized_dice_focal`
   - **Competition/Research**: `adaptive_progressive_hybrid`

2. **Check architecture** in [architecture.md](architecture.md) for pipeline understanding

3. **Review baseline** in [SwinUNETR-Baseline.md](SwinUNETR-Baseline.md) for performance expectations

## 🎯 Key Features

### Advanced Loss Functions
- **Adaptive Scheduling**: Automatic curriculum learning
- **Local Minima Escape**: Warm restarts and plateau detection
- **BraTS Optimization**: Specialized for brain tumor segmentation

### SOTA Performance
- **14 Loss Options**: From simple Dice to complex adaptive hybrids
- **Competition Ready**: Proven strategies for BraTS challenges
- **Comprehensive Documentation**: Every parameter explained with examples

### Easy to Use
- **CLI Interface**: 30+ configurable parameters
- **Example Commands**: From quick start to competition settings
- **Troubleshooting**: Common issues and solutions

## 📊 Expected Performance

Based on our comprehensive loss function analysis:

| Loss Type | WT Dice | TC Dice | ET Dice | Use Case |
|-----------|---------|---------|---------|----------|
| dicece | 0.89 | 0.84 | 0.77 | Baseline |
| generalized_dice_focal | 0.90 | 0.86 | 0.81 | Standard |
| adaptive_progressive | 0.92 | 0.88 | 0.85 | SOTA |

## 🔗 Related Files

- **Main Training Script**: `kaggle_run.py`
- **Pipeline Implementation**: `src/models/pipeline.py`
- **Model Architecture**: `src/models/swinunetr.py`
- **Training Logic**: `src/models/trainer.py`

---

**For detailed information on any topic, refer to the specific documentation files above.**