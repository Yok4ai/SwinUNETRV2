# SwinUNETR V2 Loss Functions Comprehensive Guide

This document provides a detailed overview of all loss functions available in the SwinUNETR V2 pipeline for brain tumor segmentation on BraTS datasets.

## Quick Summary

### 🎯 Recommended for Different Scenarios

| **Scenario** | **Loss Function** | **Command** |
|:-------------|:------------------|:------------|
| **Beginner/Baseline** | `dicece` | `--loss_type dicece --use_class_weights` |
| **Standard Production** | `generalized_dice_focal` | `--loss_type generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5` |
| **Competition/SOTA** | `adaptive_progressive_hybrid` | `--loss_type adaptive_progressive_hybrid --use_adaptive_scheduling` |
| **Small Lesions (ET)** | `generalized_dice_focal` | `--loss_type generalized_dice_focal --class_weights 5.0 1.0 8.0` |
| **Boundary Quality** | `hybrid_dice_hausdorff` | `--loss_type hybrid_dice_hausdorff --lambda_hausdorff 0.3` |

### 📊 Loss Function Characteristics

| **Loss Function** | **Complexity** | **Memory** | **Convergence** | **Best For** |
|:------------------|:--------------:|:----------:|:---------------:|:-------------|
| `dice` | ⚡ Simple | Low | Fast | Baseline experiments |
| `dicece` | ⚡ Simple | Low | Fast | General purpose |
| `dicefocal` | 🔶 Medium | Medium | Medium | Class imbalance |
| `generalized_dice_focal` | 🔶 Medium | Medium | Medium | Small lesions |
| `hybrid_gdl_focal_tversky` | 🔥 Complex | High | Slow | Multi-objective |
| `adaptive_progressive` | 🔥 Complex | High | Variable | Curriculum learning |

### 🚀 Quick Start Commands

```bash
# Beginner (fast training, good results)
python kaggle_run.py --loss_type dicece --use_class_weights --epochs 50

# Production (balanced performance/time)
python kaggle_run.py --loss_type generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5 --epochs 80

# Competition (maximum performance)
python kaggle_run.py --loss_type adaptive_progressive_hybrid --use_adaptive_scheduling \
  --structure_epochs 40 --boundary_epochs 70 --epochs 120 --class_weights 4.0 1.0 6.0
```

### 📈 Loss Function Categories

| **Category** | **Functions** | **Use Case** |
|:-------------|:--------------|:-------------|
| **Basic** | `dice`, `dicece`, `dicefocal`, `focal`, `tversky`, `hausdorff` | Simple, reliable training |
| **Advanced** | `generalized_dice`, `generalized_dice_focal` | Class imbalance handling |
| **Hybrid** | `hybrid_gdl_focal_tversky`, `hybrid_dice_hausdorff` | Multi-objective optimization |
| **Adaptive** | `adaptive_structure_boundary`, `adaptive_progressive_hybrid`, `adaptive_complexity_cascade`, `adaptive_dynamic_hybrid` | Curriculum learning, SOTA performance |

### ⚙️ Key Parameters

| **Parameter** | **Typical Values** | **Effect** |
|:--------------|:-------------------|:-----------|
| `class_weights` | `[4.0, 1.0, 6.0]` | Balance TC/WT/ET importance |
| `focal_gamma` | `2.0-3.0` | Focus on hard examples |
| `tversky_alpha` | `0.3` (sensitivity) / `0.7` (precision) | Control FN/FP trade-off |
| `learning_rate` | `1e-4` to `1e-3` | Training speed/stability |

## Table of Contents
- [Basic Loss Functions](#basic-loss-functions)
- [Hybrid Loss Functions](#hybrid-loss-functions)
- [Adaptive Loss Functions](#adaptive-loss-functions)
- [SOTA Strategies & Local Minima Escape](#sota-strategies--local-minima-escape)
- [Parameter Configuration](#parameter-configuration)
- [BraTS Dataset Recommendations](#brats-dataset-recommendations)
- [Performance Comparison](#performance-comparison)
- [Usage Examples](#usage-examples)

---

## Basic Loss Functions

### 1. Dice Loss (`dice`)

**What it does:**
- Measures overlap between predicted and ground truth segmentation masks
- Based on the Sørensen-Dice coefficient: `2 * |A ∩ B| / (|A| + |B|)`
- Focuses on maximizing region overlap

**Mathematical Formula:**
```
Dice Loss = 1 - (2 * TP) / (2 * TP + FP + FN)
```

**Parameters:**
- `smooth_nr=0`: Numerator smoothing
- `smooth_dr=1e-5`: Denominator smoothing to avoid division by zero
- `squared_pred=True`: Use squared predictions for smoother gradients
- `weight`: Class weights for handling imbalanced data

**When to use:**
- ✅ **Baseline experiments** - Simple and reliable
- ✅ **Balanced datasets** - Works well when classes are roughly equal
- ✅ **Initial prototyping** - Good starting point for new architectures
- ❌ Avoid for highly imbalanced classes (e.g., small ET regions)

**BraTS Performance:**
- **TC (Tumor Core)**: Good performance (0.85+ Dice)
- **WT (Whole Tumor)**: Excellent performance (0.90+ Dice)
- **ET (Enhancing Tumor)**: Moderate performance (0.75+ Dice)

**Usage:**
```bash
python kaggle_run.py --loss_type dice --use_class_weights
```

---

### 2. Dice + Cross-Entropy Loss (`dicece`)

**What it does:**
- Combines Dice loss with Cross-Entropy loss
- Dice handles region overlap, CE handles pixel-wise classification
- Provides balanced optimization for both region-level and pixel-level accuracy

**Mathematical Formula:**
```
DiceCE Loss = λ_dice * Dice_Loss + λ_ce * CrossEntropy_Loss
```

**Parameters:**
- `lambda_dice`: Weight for Dice component (default: 1.0)
- `lambda_ce`: Weight for Cross-Entropy component (default: 1.0)
- All Dice parameters apply

**When to use:**
- ✅ **Standard segmentation tasks** - Most widely used combination
- ✅ **Moderate class imbalance** - CE helps with pixel-level accuracy
- ✅ **Multi-class segmentation** - Handles 3-class BraTS well
- ✅ **Production models** - Stable and reliable performance

**BraTS Performance:**
- **Overall**: Balanced performance across all tumor regions
- **Best for**: General-purpose brain tumor segmentation
- **Training stability**: High - converges reliably

**Usage:**
```bash
python kaggle_run.py --loss_type dicece --lambda_dice 1.0 --use_class_weights
```

---

### 3. Dice + Focal Loss (`dicefocal`)

**What it does:**
- Combines Dice loss with Focal loss
- Focal loss focuses on hard-to-classify pixels
- Excellent for handling class imbalance and boundary refinement

**Mathematical Formula:**
```
DiceFocal Loss = λ_dice * Dice_Loss + λ_focal * Focal_Loss
Focal_Loss = -α * (1-p)^γ * log(p)
```

**Parameters:**
- `lambda_dice`: Weight for Dice component (default: 1.0)
- `lambda_focal`: Weight for Focal component (default: 1.0)
- `focal_gamma`: Focusing parameter γ (default: 2.0)
- `focal_alpha`: Class weighting parameter α (default: None)

**When to use:**
- ✅ **Class imbalance** - Focal loss handles rare classes well
- ✅ **Boundary refinement** - Focuses on hard pixels at boundaries
- ✅ **ET segmentation** - Excellent for small enhancing tumor regions
- ✅ **Fine-tuning** - Use after initial training with simpler losses

**BraTS Performance:**
- **ET (Enhancing Tumor)**: Excellent performance (0.80+ Dice)
- **Boundary quality**: Superior edge definition
- **Small lesions**: Better detection of tiny tumor regions

**Usage:**
```bash
python kaggle_run.py --loss_type dicefocal --focal_gamma 2.0 --lambda_dice 1.0 --lambda_focal 1.0 --use_class_weights
```

---

### 4. Generalized Dice Loss (`generalized_dice`)

**What it does:**
- Extension of Dice loss that handles class imbalance internally
- Automatically weights classes based on their inverse frequency
- No need for manual class weight tuning

**Mathematical Formula:**
```
GDL = 1 - 2 * Σ(w_i * Σ(r_i * p_i)) / Σ(w_i * Σ(r_i + p_i))
where w_i = 1 / (Σ(r_i))^2  # inverse frequency weighting
```

**Parameters:**
- `w_type`: Weighting strategy ('square', 'simple', 'uniform')
- `smooth_nr`, `smooth_dr`: Smoothing parameters

**When to use:**
- ✅ **Severe class imbalance** - Built-in automatic reweighting
- ✅ **No manual tuning** - Automatically adapts to data distribution
- ✅ **ET-dominant tasks** - Excellent for rare tumor components
- ❌ Avoid when classes are balanced

**BraTS Performance:**
- **ET**: Outstanding performance for small lesions
- **TC**: Good performance with automatic weighting
- **Imbalanced datasets**: Superior to standard Dice

**Usage:**
```bash
python kaggle_run.py --loss_type generalized_dice --gdl_weight_type square
```

---

### 5. Generalized Dice + Focal Loss (`generalized_dice_focal`)

**What it does:**
- Combines GDL's automatic class balancing with Focal loss's hard example mining
- Best of both worlds for imbalanced segmentation
- Excellent for BraTS datasets with rare ET regions

**Parameters:**
- `lambda_gdl`: Weight for GDL component (default: 1.0)
- `lambda_focal`: Weight for Focal component (default: 1.0)
- `focal_gamma`: Focusing parameter
- `gdl_weight_type`: GDL weighting strategy

**When to use:**
- ✅ **Highly imbalanced BraTS data** - Handles ET regions excellently
- ✅ **Challenging boundary cases** - Superior edge definition
- ✅ **Competition settings** - Often achieves best performance
- ✅ **Small lesion detection** - Outstanding for tiny tumor regions

**BraTS Performance:**
- **ET**: Best-in-class performance (0.82+ Dice)
- **Overall**: Top-tier performance across all metrics
- **Boundary quality**: Excellent edge preservation

**Usage:**
```bash
python kaggle_run.py --loss_type generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5 --focal_gamma 2.0
```

---

### 6. Focal Loss (`focal`)

**What it does:**
- Focuses training on hard-to-classify examples
- Downweights easy examples during training
- Primarily designed for classification but adapted for segmentation

**Mathematical Formula:**
```
Focal Loss = -α * (1-p)^γ * log(p)
```

**Parameters:**
- `gamma`: Focusing parameter (higher = more focus on hard examples)
- `alpha`: Class balancing parameter

**When to use:**
- ✅ **Hard example mining** - When model struggles with specific regions
- ✅ **Boundary improvement** - Focuses on difficult boundary pixels
- ❌ Standalone use not recommended for BraTS
- ✅ **Component in hybrid losses** - Works well in combinations

**Usage:**
```bash
python kaggle_run.py --loss_type focal --focal_gamma 2.0 --use_class_weights
```

---

### 7. Tversky Loss (`tversky`)

**What it does:**
- Generalization of Dice loss with asymmetric weighting
- Controls trade-off between false positives and false negatives
- Excellent for precision/recall balance tuning

**Mathematical Formula:**
```
Tversky Loss = 1 - TP / (TP + α*FN + β*FP)
```

**Parameters:**
- `tversky_alpha`: Weight for false negatives (default: 0.5)
- `tversky_beta`: Weight for false positives (default: 0.5)

**When to use:**
- ✅ **Precision-recall tuning** - α > β favors precision, α < β favors recall
- ✅ **Medical applications** - When false negatives are more critical
- ✅ **Small object detection** - Set α < β to increase sensitivity
- ✅ **Custom requirements** - When specific precision/recall balance needed

**BraTS Tuning:**
- **High Sensitivity (α=0.3, β=0.7)**: Better ET detection, more false positives
- **High Precision (α=0.7, β=0.3)**: Cleaner boundaries, risk missing small lesions
- **Balanced (α=0.5, β=0.5)**: Equivalent to Dice loss

**Usage:**
```bash
python kaggle_run.py --loss_type tversky --tversky_alpha 0.3 --tversky_beta 0.7
```

---

### 8. Hausdorff Distance Loss (`hausdorff`)

**What it does:**
- Measures maximum distance between predicted and true boundaries
- Focuses on boundary accuracy and shape preservation
- Excellent for applications requiring precise edge definition

**Mathematical Formula:**
```
Hausdorff Loss = max(max(d(a,B)), max(d(b,A)))
where d(x,S) is distance from point x to set S
```

**Parameters:**
- `hausdorff_alpha`: Scaling parameter (default: 2.0)
- `include_background`: Whether to include background class

**When to use:**
- ✅ **Boundary quality critical** - Medical imaging applications
- ✅ **Shape preservation** - When tumor shape matters
- ✅ **Fine-tuning phase** - After initial convergence with other losses
- ❌ Standalone use challenging - better in hybrid combinations
- ✅ **Radiological assessment** - When boundary accuracy is paramount

**BraTS Performance:**
- **Boundary quality**: Outstanding edge definition
- **Shape preservation**: Excellent tumor contour accuracy
- **Training stability**: Requires careful tuning

**Usage:**
```bash
python kaggle_run.py --loss_type hausdorff --hausdorff_alpha 2.0
```

---

## Hybrid Loss Functions

### 9. Hybrid GDL + Focal + Tversky (`hybrid_gdl_focal_tversky`)

**What it does:**
- Combines three powerful loss functions for comprehensive optimization
- GDL handles class imbalance, Focal focuses on hard examples, Tversky balances precision/recall
- Most sophisticated non-adaptive loss combination

**Mathematical Formula:**
```
Total Loss = λ_gdl * GDL_Loss + λ_focal * Focal_Loss + λ_tversky * Tversky_Loss
```

**Parameters:**
- `gdl_lambda`: Weight for GDL component
- `lambda_focal`: Weight for Focal component  
- `lambda_tversky`: Weight for Tversky component
- All individual loss parameters apply

**When to use:**
- ✅ **Competition settings** - Maximum performance for challenges
- ✅ **Research experiments** - Comprehensive optimization
- ✅ **Challenging datasets** - Complex tumor presentations
- ✅ **Final model training** - After hyperparameter optimization

**Recommended Parameters:**
```bash
# Balanced approach
--gdl_lambda 1.0 --lambda_focal 0.5 --lambda_tversky 0.3

# ET-focused (small lesions)
--gdl_lambda 1.2 --lambda_focal 0.8 --lambda_tversky 0.4 --tversky_alpha 0.3

# Boundary-focused  
--gdl_lambda 0.8 --lambda_focal 1.0 --lambda_tversky 0.6 --focal_gamma 3.0
```

**Usage:**
```bash
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky \
  --gdl_lambda 1.0 --lambda_focal 0.5 --lambda_tversky 0.3 \
  --focal_gamma 2.0 --tversky_alpha 0.3 --tversky_beta 0.7
```

---

### 10. Hybrid Dice + Hausdorff (`hybrid_dice_hausdorff`)

**What it does:**
- Combines region overlap optimization (Dice) with boundary quality (Hausdorff)
- Balanced approach for both volume accuracy and shape preservation
- Excellent for medical applications requiring both metrics

**Mathematical Formula:**
```
Total Loss = λ_dice * Dice_Loss + λ_hausdorff * Hausdorff_Loss
```

**When to use:**
- ✅ **Medical imaging** - When both volume and boundary matter
- ✅ **Shape-critical applications** - Tumor morphology important
- ✅ **Balanced optimization** - Good compromise between metrics
- ✅ **Clinical validation** - Matches radiological assessment criteria

**Recommended Parameters:**
```bash
# Standard balance
--lambda_dice 1.0 --lambda_hausdorff 0.1

# Boundary-focused
--lambda_dice 0.8 --lambda_hausdorff 0.3

# Volume-focused  
--lambda_dice 1.2 --lambda_hausdorff 0.05
```

**Usage:**
```bash
python kaggle_run.py --loss_type hybrid_dice_hausdorff \
  --lambda_dice 1.0 --lambda_hausdorff 0.1 --hausdorff_alpha 2.0
```

---

## Adaptive Loss Functions

### 11. Adaptive Structure-Boundary (`adaptive_structure_boundary`)

**What it does:**
- **Early Training**: Focuses on structure learning with Dice loss
- **Late Training**: Shifts to boundary refinement with Focal loss  
- **Progressive Difficulty**: Smooth transition between learning phases
- **Dynamic Weighting**: Weights adapt based on epoch and schedule type

**Learning Phases:**
1. **Structure Phase**: High Dice weight, low Focal weight
2. **Transition Phase**: Gradual weight shifting
3. **Boundary Phase**: Low Dice weight, high Focal weight

**Schedule Types:**
- **Linear**: Smooth linear transition between phases
- **Exponential**: Rapid early transition, gradual later
- **Cosine**: Sinusoidal transition for smooth optimization

**When to use:**
- ✅ **Long training runs** - 100+ epochs for full benefit
- ✅ **From-scratch training** - Best for training without pretrained weights
- ✅ **Challenging boundaries** - When edge quality is critical
- ✅ **Curriculum learning** - Progressive difficulty approach

**Parameters:**
- `adaptive_schedule_type`: 'linear', 'exponential', 'cosine'
- `schedule_start_epoch`: When to begin adaptive scheduling
- `min_loss_weight`: Minimum weight for any component
- `max_loss_weight`: Maximum weight for any component

**Usage:**
```bash
python kaggle_run.py --loss_type adaptive_structure_boundary \
  --use_adaptive_scheduling --adaptive_schedule_type cosine \
  --schedule_start_epoch 15 --min_loss_weight 0.2 --max_loss_weight 1.5
```

---

### 12. Adaptive Progressive Hybrid (`adaptive_progressive_hybrid`)

**What it does:**
- **Phase 1** (0-structure_epochs): Pure Dice for basic structure learning
- **Phase 2** (structure-boundary_epochs): Dice + Focal for boundary refinement  
- **Phase 3** (boundary_epochs+): Dice + Focal + Tversky for precision/recall balance
- **Progressive Complexity**: Gradually adds loss components with stable gradients

**Training Progression:**
```
Epochs 0-30:    Dice (structure learning)
Epochs 30-50:   Dice + Focal (boundary refinement)  
Epochs 50+:     Dice + Focal + Tversky (precision/recall balance)
```

**Why Tversky instead of Hausdorff:**
- **Stable gradients**: No discontinuities or gradient instability
- **Tunable balance**: α/β parameters control precision vs recall trade-off
- **Better convergence**: Avoids performance drops common with Hausdorff in Phase 3
- **Complementary optimization**: Focuses on different aspects than Dice+Focal

**When to use:**
- ✅ **Extended training** - 120+ epochs recommended
- ✅ **Complex datasets** - Multiple tumor types and challenging cases
- ✅ **Research settings** - When training time is not a constraint
- ✅ **Maximum quality** - When best possible results are needed

**Parameters:**
- `structure_epochs`: Duration of structure learning phase (default: 30)
- `boundary_epochs`: When to add boundary refinement (default: 50)  
- `schedule_start_epoch`: When to begin scheduling (default: 10)
- `tversky_alpha`: Controls false negative penalty (default: 0.5)
- `tversky_beta`: Controls false positive penalty (default: 0.5)

**Usage:**
```bash
python kaggle_run.py --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling --structure_epochs 40 --boundary_epochs 70 \
  --schedule_start_epoch 10 --tversky_alpha 0.3 --tversky_beta 0.7
```

---

### 13. Adaptive Complexity Cascade (`adaptive_complexity_cascade`)

**What it does:**
- **Stage 1** (0-25%): Pure Dice for basic learning
- **Stage 2** (25-50%): Dice + DiceCE for improved classification
- **Stage 3** (50-75%): Dice + DiceCE + DiceFocal for hard examples
- **Stage 4** (75-100%): All losses for comprehensive optimization

**Cascading Strategy:**
- Gradually introduces more sophisticated loss components
- Each stage builds upon previous learning
- Smooth transitions between complexity levels

**When to use:**
- ✅ **Curriculum learning** - Systematic complexity introduction
- ✅ **Stable training** - Reduces optimization difficulties
- ✅ **BraTS competitions** - Comprehensive approach for best results
- ✅ **Methodical optimization** - When systematic approach preferred

**Usage:**
```bash
python kaggle_run.py --loss_type adaptive_complexity_cascade \
  --use_adaptive_scheduling --adaptive_schedule_type linear
```

---

### 14. Adaptive Dynamic Hybrid (`adaptive_dynamic_hybrid`)

**What it does:**
- **Performance-Based Adaptation**: Adjusts weights based on validation Dice scores
- **Poor Performance** (<0.7 Dice): Focus on structure with GDL
- **Moderate Performance** (0.7-0.85 Dice): Balanced multi-loss approach
- **Good Performance** (>0.85 Dice): Focus on fine details and boundaries

**Dynamic Strategy:**
```python
if val_dice < 0.7:    # Focus on structure
    weights = {'gdl': 2.0, 'focal': 0.1, 'tversky': 0.1, 'hausdorff': 0.0}
elif val_dice < 0.85: # Balanced approach  
    weights = {'gdl': 1.6, 'focal': 1.2, 'tversky': 1.2, 'hausdorff': 0.1}
else:                 # Focus on details
    weights = {'gdl': 1.2, 'focal': 2.0, 'tversky': 1.6, 'hausdorff': 1.4}
```

**When to use:**
- ✅ **Adaptive optimization** - Automatically adjusts to model performance
- ✅ **Unknown datasets** - When optimal strategy is unclear
- ✅ **Automatic tuning** - Reduces manual hyperparameter selection
- ✅ **Online learning** - Adapts during training without intervention

**Usage:**
```bash
python kaggle_run.py --loss_type adaptive_dynamic_hybrid \
  --use_adaptive_scheduling --schedule_start_epoch 20
```

---

## Parameter Configuration Guide

### Class Weights for BraTS
```python
# Conservative (balanced)
class_weights = [3.0, 1.0, 5.0]  # TC, WT, ET

# Aggressive ET focus  
class_weights = [4.0, 1.0, 8.0]  # Higher ET weight

# WT focus (large tumors)
class_weights = [2.0, 0.5, 4.0]  # Lower WT weight
```

### Focal Loss Parameters
```python
# Standard settings
focal_gamma = 2.0    # Good balance
focal_alpha = None   # Use class weights instead

# Hard example focus
focal_gamma = 3.0    # More aggressive focusing

# Gentle focusing
focal_gamma = 1.5    # Less aggressive
```

### Tversky Parameters
```python
# High sensitivity (catch all tumors)
tversky_alpha = 0.3, tversky_beta = 0.7

# High precision (clean boundaries)  
tversky_alpha = 0.7, tversky_beta = 0.3

# Balanced (equivalent to Dice)
tversky_alpha = 0.5, tversky_beta = 0.5
```

---

## BraTS Dataset Recommendations

### BraTS 2021 vs 2023
- **BraTS 2021**: More conservative loss weights, focus on stability
- **BraTS 2023**: Can use more aggressive parameters, better data quality

### Dataset Size Considerations
- **Small datasets** (<100 cases): Use simpler losses (dice, dicece)
- **Medium datasets** (100-300 cases): Hybrid losses work well
- **Large datasets** (300+ cases): Adaptive losses show full potential

### Tumor Type Focus

#### Whole Tumor (WT) Optimization
```bash
python kaggle_run.py --loss_type dicece --class_weights 2.0 0.5 3.0
```

#### Tumor Core (TC) Optimization  
```bash
python kaggle_run.py --loss_type generalized_dice --class_weights 4.0 1.0 4.0
```

#### Enhancing Tumor (ET) Optimization
```bash
python kaggle_run.py --loss_type generalized_dice_focal \
  --class_weights 5.0 1.0 8.0 --focal_gamma 2.5
```

---

## SOTA Strategies & Local Minima Escape

### Understanding Local Minima in Medical Segmentation

Medical image segmentation models often get trapped in local minima, leading to suboptimal performance. This is particularly common in BraTS datasets due to:

- **Class imbalance**: ET regions are 100-1000x smaller than background
- **Boundary complexity**: Irregular tumor shapes with fuzzy boundaries  
- **Inter-tumor variability**: Different tumor types require different optimization strategies
- **Limited data**: Medical datasets are smaller than natural image datasets

### 🎯 SOTA Hybrid Dice Strategies

#### 1. **Progressive Loss Scheduling (Recommended for SOTA)**

**Strategy**: Start with structure learning, progressively add complexity
```bash
# Phase 1: Structure Foundation (Epochs 0-30)
python kaggle_run.py --loss_type dice --learning_rate 1e-3 --epochs 30

# Phase 2: Add Boundary Refinement (Epochs 30-60)  
python kaggle_run.py --loss_type dicefocal --learning_rate 5e-4 --epochs 60 \
  --resume_from_checkpoint checkpoints/epoch_30.ckpt

# Phase 3: Fine Details (Epochs 60-100)
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky --learning_rate 1e-4 --epochs 100 \
  --gdl_lambda 1.0 --lambda_focal 0.8 --lambda_tversky 0.6 \
  --resume_from_checkpoint checkpoints/epoch_60.ckpt
```

**Why this works:**
- Prevents early convergence to poor local minima
- Each phase builds upon previous learning
- Gradually increases optimization difficulty

#### 2. **Adaptive Restart Strategy**

**Strategy**: Automatically restart with different loss when stuck
```bash
# Monitor validation plateau and switch losses
python kaggle_run.py --loss_type adaptive_dynamic_hybrid \
  --use_adaptive_scheduling --schedule_start_epoch 20 \
  --min_loss_weight 0.05 --max_loss_weight 3.0
```

**Implementation Logic:**
```python
# Pseudo-code for adaptive restart
if val_dice_plateau > 10_epochs:
    if current_loss == 'dice':
        switch_to('generalized_dice_focal')
    elif current_loss == 'generalized_dice_focal':
        switch_to('hybrid_gdl_focal_tversky')
```

#### 3. **Multi-Scale Loss Ensemble**

**Strategy**: Train multiple models with different loss functions, ensemble predictions
```bash
# Model 1: Structure specialist
python kaggle_run.py --loss_type generalized_dice --class_weights 2.0 1.0 4.0

# Model 2: Boundary specialist  
python kaggle_run.py --loss_type dicefocal --focal_gamma 3.0 --class_weights 4.0 1.0 8.0

# Model 3: Balanced hybrid
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky \
  --gdl_lambda 1.0 --lambda_focal 0.7 --lambda_tversky 0.5

# Ensemble predictions using weighted averaging
```

### 🚀 Local Minima Escape Techniques

#### Technique 1: **Loss Function Annealing**
```bash
# Start aggressive, gradually moderate
python kaggle_run.py --loss_type adaptive_structure_boundary \
  --use_adaptive_scheduling --adaptive_schedule_type exponential \
  --min_loss_weight 0.1 --max_loss_weight 5.0  # High initial range
```

#### Technique 2: **Cyclical Loss Weights**
```python
# Implement cyclical weighting (add to pipeline.py)
def get_cyclical_weights(self, epoch):
    cycle_length = 20
    cycle_position = epoch % cycle_length
    
    # Sine wave between min and max weights
    weight_factor = 0.5 * (1 + math.sin(2 * math.pi * cycle_position / cycle_length))
    dice_weight = self.min_loss_weight + weight_factor * (self.max_loss_weight - self.min_loss_weight)
    focal_weight = self.max_loss_weight - weight_factor * (self.max_loss_weight - self.min_loss_weight)
    
    return dice_weight, focal_weight
```

#### Technique 3: **Warm Restart with Loss Switching**
```bash
# Restart training with different loss every 40 epochs
for restart in range(3):
    python kaggle_run.py --loss_type dice --epochs 40 --learning_rate 1e-3
    python kaggle_run.py --loss_type dicefocal --epochs 40 --learning_rate 5e-4 \
      --resume_from_checkpoint checkpoints/restart_${restart}_dice.ckpt
    python kaggle_run.py --loss_type hybrid_gdl_focal_tversky --epochs 40 --learning_rate 1e-4 \
      --resume_from_checkpoint checkpoints/restart_${restart}_dicefocal.ckpt
done
```

### 🏆 Competition-Winning Strategies

#### Strategy 1: **The BraTS Champion Approach**
```bash
# Stage 1: Foundation (0-40 epochs)
python kaggle_run.py --loss_type generalized_dice \
  --learning_rate 2e-3 --class_weights 3.0 1.0 6.0 --epochs 40

# Stage 2: Boundary Focus (40-80 epochs)  
python kaggle_run.py --loss_type generalized_dice_focal \
  --learning_rate 1e-3 --focal_gamma 2.5 --epochs 80 \
  --gdl_lambda 1.2 --lambda_focal 0.8 --class_weights 4.0 1.0 8.0

# Stage 3: Fine-tuning (80-120 epochs)
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky \
  --learning_rate 3e-4 --epochs 120 \
  --gdl_lambda 1.0 --lambda_focal 1.0 --lambda_tversky 0.7 \
  --tversky_alpha 0.3 --tversky_beta 0.7 --focal_gamma 3.0

# Stage 4: Polish (120-150 epochs)
python kaggle_run.py --loss_type hybrid_dice_hausdorff \
  --learning_rate 1e-4 --epochs 150 \
  --lambda_dice 0.8 --lambda_hausdorff 0.4 --hausdorff_alpha 2.5
```

#### Strategy 2: **The Robust Ensemble**
```bash
# Train 5 models with different loss strategies
models=(
  "generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5"
  "hybrid_gdl_focal_tversky --gdl_lambda 1.2 --lambda_focal 0.8 --lambda_tversky 0.6"
  "adaptive_progressive_hybrid --structure_epochs 35 --boundary_epochs 65"
  "dicefocal --focal_gamma 2.5 --lambda_dice 1.0 --lambda_focal 1.2"  
  "adaptive_dynamic_hybrid --schedule_start_epoch 15"
)

for i in "${!models[@]}"; do
  python kaggle_run.py --loss_type ${models[$i]} \
    --epochs 100 --learning_rate 1e-4 --class_weights 4.0 1.0 6.0 \
    --output_dir model_$i
done

# Ensemble with weighted voting based on validation performance
```

### 🔬 Advanced Optimization Tricks

#### 1. **Loss Landscape Smoothing**
```bash
# Use larger batch sizes or gradient accumulation for smoother loss landscape
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky \
  --batch_size 4 --accumulate_grad_batches 2  # Effective batch size = 8
```

#### 2. **Stochastic Weight Averaging (SWA)**
```bash
# Apply SWA in final training phase
python kaggle_run.py --loss_type generalized_dice_focal \
  --epochs 120 --swa_start_epoch 80 --swa_lr 1e-5
```

#### 3. **Multi-Objective Optimization**
```python
# Custom loss that balances multiple objectives
def multi_objective_loss(outputs, labels):
    dice_loss = compute_dice_loss(outputs, labels)
    boundary_loss = compute_boundary_loss(outputs, labels)  
    shape_loss = compute_shape_loss(outputs, labels)
    
    # Dynamic weighting based on current performance
    weights = get_dynamic_weights(current_metrics)
    
    return (weights['dice'] * dice_loss + 
            weights['boundary'] * boundary_loss + 
            weights['shape'] * shape_loss)
```

### 📊 Performance Monitoring & Escape Detection

#### Early Warning Signs of Local Minima:
1. **Validation plateau** for >15 epochs
2. **Loss oscillation** without improvement
3. **Metric imbalance** (good WT, poor ET)
4. **Gradient norms** approaching zero

#### Automated Escape Protocol:
```python
class LocalMinimaDetector:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.plateau_count = 0
        self.best_metric = -1
        
    def detect_plateau(self, current_metric):
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.plateau_count = 0
        else:
            self.plateau_count += 1
            
        return self.plateau_count >= self.patience
        
    def suggest_escape_strategy(self, current_loss_type):
        strategies = {
            'dice': 'generalized_dice_focal',
            'dicece': 'hybrid_gdl_focal_tversky', 
            'generalized_dice': 'adaptive_progressive_hybrid'
        }
        return strategies.get(current_loss_type, 'adaptive_dynamic_hybrid')
```

### 💡 SOTA Hybrid Dice Recommendations

#### For Competition/Research (Maximum Performance):
```bash
# The Ultimate SOTA Pipeline - Now with stable Tversky Phase 3
python kaggle_run.py --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling \
  --structure_epochs 45 --boundary_epochs 75 --schedule_start_epoch 15 \
  --adaptive_schedule_type cosine \
  --min_loss_weight 0.1 --max_loss_weight 2.5 \
  --epochs 150 --learning_rate 1e-4 \
  --class_weights 5.0 1.0 10.0 \
  --focal_gamma 2.8 --tversky_alpha 0.25 --tversky_beta 0.75 \
  --use_class_weights --use_modality_attention
```

#### For Fast Convergence (Time-Constrained):
```bash
# Aggressive early convergence strategy
python kaggle_run.py --loss_type generalized_dice_focal \
  --gdl_lambda 1.5 --lambda_focal 1.0 --focal_gamma 3.0 \
  --learning_rate 2e-3 --warmup_epochs 5 \
  --class_weights 6.0 1.0 12.0 --epochs 60
```

#### For Robust Performance (Production):
```bash
# Stable, reliable training
python kaggle_run.py --loss_type hybrid_gdl_focal_tversky \
  --gdl_lambda 1.0 --lambda_focal 0.7 --lambda_tversky 0.5 \
  --learning_rate 8e-4 --warmup_epochs 12 \
  --class_weights 4.0 1.0 7.0 --epochs 100
```

### 🎓 Key Takeaways for Local Minima Avoidance:

1. **Never start with complex losses** - Begin simple, add complexity
2. **Use progressive scheduling** - Adaptive losses are game-changers
3. **Monitor multiple metrics** - Not just overall Dice score
4. **Implement escape protocols** - Automatic loss switching when stuck
5. **Ensemble different strategies** - Combine multiple loss approaches
6. **Validate extensively** - Use proper cross-validation and test sets

---

## Performance Comparison

### Computational Cost (Training Time)
1. **Fastest**: `dice`, `dicece` 
2. **Medium**: `dicefocal`, `generalized_dice`
3. **Slow**: `hausdorff`, hybrid losses
4. **Slowest**: Adaptive losses (worth the time)

### Memory Usage
1. **Lowest**: Basic losses
2. **Medium**: Hybrid losses  
3. **Highest**: Adaptive losses (multiple loss computations)

### Convergence Speed
1. **Fastest**: `dice`, `dicece`
2. **Medium**: `generalized_dice`, `dicefocal`
3. **Slower**: Hybrid and adaptive losses (but better final performance)

### Performance Characteristics

- **Simple losses** (dice, dicece): Fast convergence, good baseline
- **Focal-based losses**: Better for imbalanced classes, especially ET regions  
- **Hybrid losses**: Balance multiple objectives, slower but more comprehensive
- **Adaptive losses**: Best potential performance with proper tuning and time

---

## Usage Examples

### Quick Start (Balanced Performance)
```bash
python kaggle_run.py --loss_type dicece --use_class_weights --epochs 50
```

### Competition Setting (Maximum Performance)
```bash
python kaggle_run.py --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling --structure_epochs 40 --boundary_epochs 70 \
  --epochs 120 --use_class_weights --class_weights 4.0 1.0 6.0
```

### ET-Focused (Small Lesion Detection)
```bash
python kaggle_run.py --loss_type generalized_dice_focal \
  --gdl_lambda 1.0 --lambda_focal 0.8 --focal_gamma 2.5 \
  --use_class_weights --class_weights 5.0 1.0 8.0
```

### Boundary Quality Focus
```bash
python kaggle_run.py --loss_type hybrid_dice_hausdorff \
  --lambda_dice 0.8 --lambda_hausdorff 0.3 --hausdorff_alpha 2.0
```

### Research/Experimental
```bash
python kaggle_run.py --loss_type adaptive_dynamic_hybrid \
  --use_adaptive_scheduling --schedule_start_epoch 15 \
  --min_loss_weight 0.1 --max_loss_weight 2.5
```

---

## Best Practices

### 1. Loss Selection Strategy
1. **Start Simple**: Begin with `dicece` for baseline
2. **Add Complexity**: Move to `generalized_dice_focal` for imbalanced data
3. **Fine-tune**: Use hybrid losses for specific requirements
4. **Optimize**: Apply adaptive losses for maximum performance

### 2. Hyperparameter Tuning Order
1. **Learning rate and batch size** first
2. **Class weights** based on dataset distribution  
3. **Loss-specific parameters** (gamma, alpha, beta)
4. **Loss combination weights** (lambdas)
5. **Adaptive scheduling parameters** last

### 3. Training Strategy
1. **Warmup epochs**: Always use 10-15 warmup epochs
2. **Early stopping**: Monitor validation Dice with patience=15
3. **Learning rate scheduling**: Cosine annealing works well
4. **Mixed precision**: Always enable for faster training

### 4. Validation Strategy
- Use 5-fold cross-validation for robust results
- Monitor all metrics: Dice, IoU, Hausdorff, Precision, Recall
- Save checkpoints based on mean Dice across all tumor regions
- Validate on held-out test set for final model selection

---

## Troubleshooting

### Common Issues

**Loss not decreasing:**
- Check learning rate (try 1e-4 to 1e-3)
- Verify class weights are appropriate
- Ensure data normalization is correct

**NaN losses:**
- Reduce learning rate
- Check for division by zero in custom losses
- Verify input data ranges

**Poor boundary quality:**
- Use Focal or Hausdorff loss components
- Increase focal_gamma parameter
- Add boundary-focused data augmentation

**Missing small lesions:**
- Increase class weights for ET
- Use Tversky loss with alpha < beta
- Apply Generalized Dice Loss

**Overfitting:**
- Add regularization (weight decay)
- Use data augmentation
- Implement early stopping
- Reduce model complexity

---

## Conclusion

The SwinUNETR V2 pipeline offers a comprehensive suite of loss functions ranging from simple baselines to sophisticated adaptive strategies. For BraTS datasets:

- **Beginners**: Start with `dicece`
- **Standard Use**: `generalized_dice_focal` 
- **Competition/Research**: `adaptive_progressive_hybrid`
- **Specific Requirements**: Choose hybrid combinations

The adaptive loss functions represent the state-of-the-art in medical image segmentation, providing automatic curriculum learning and performance-based optimization that eliminates much of the manual hyperparameter tuning traditionally required.

Remember: The best loss function depends on your specific dataset, computational resources, and performance requirements. Always validate your choice with proper cross-validation and held-out test sets.