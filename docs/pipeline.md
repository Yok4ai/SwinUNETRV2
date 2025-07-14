# SwinUNETR++ Pipeline Implementation

This document provides a comprehensive overview of the `BrainTumorSegmentation` pipeline implementation in `src/models/pipeline.py`.

## Table of Contents
- [Pipeline Overview](#pipeline-overview)
- [Architecture Components](#architecture-components)
- [Loss Function System](#loss-function-system)
- [Adaptive Scheduling](#adaptive-scheduling)
- [Training Pipeline](#training-pipeline)
- [Metrics and Logging](#metrics-and-logging)
- [Optimization](#optimization)

---

## Pipeline Overview

The `BrainTumorSegmentation` class is a PyTorch Lightning module that wraps the SwinUNETR++ model with comprehensive loss function optimization, adaptive scheduling, and local minima escape strategies.

### Key Features
- **14 Loss Functions**: From basic Dice to complex adaptive hybrids
- **Adaptive Scheduling**: Dynamic loss weight transitions
- **Local Minima Escape**: Warm restarts and plateau detection
- **Comprehensive Metrics**: Dice, IoU, Hausdorff, precision, recall, F1
- **WandB Integration**: Automatic logging with visualization

---

## Architecture Components

### 1. Modality Attention Module

```python
class ModalityAttentionModule(nn.Module):
    """
    Learns importance weights for each MRI modality channel.
    Combines channel and spatial attention mechanisms.
    """
```

**Features:**
- **Channel Attention**: Global average/max pooling + shared MLP
- **Spatial Attention**: Conv3D with 7x7x7 kernel for spatial relationships
- **Residual Connection**: Preserves original input information
- **Configurable**: Reduction ratio parameter for complexity control

**When to Use:**
- Enable with `--use_modality_attention` flag
- Beneficial when different MRI modalities have varying importance
- Adds ~5-10% computational overhead but can improve performance

### 2. SwinUNETR++ Model

The core architecture uses either:
- **Standard SwinUNETR**: MONAI's implementation
- **Enhanced SwinUNETR+**: Custom implementation with multi-scale attention

**Configuration Parameters:**
```python
feature_size=48        # Base feature dimension
depths=(2, 2, 2, 2)   # Transformer depths per stage
num_heads=(3, 6, 12, 24)  # Attention heads per stage
use_v2=True           # Enable SwinTransformer V2 features
```

---

## Loss Function System

### Basic Loss Functions

#### 1. Dice Loss
```python
self.dice_loss = DiceLoss(
    smooth_nr=0, smooth_dr=1e-5, squared_pred=True, 
    to_onehot_y=False, sigmoid=True, weight=class_weights
)
```
- **Use Case**: Baseline experiments, balanced datasets
- **Pros**: Simple, reliable, good starting point
- **Cons**: Struggles with class imbalance

#### 2. DiceCE Loss
```python
self.dicece_loss = DiceCELoss(
    lambda_dice=1.0, lambda_ce=1.0, weight=class_weights
)
```
- **Use Case**: Standard segmentation, moderate imbalance
- **Pros**: Balanced pixel and region optimization
- **Cons**: May not handle severe imbalance

#### 3. DiceFocal Loss
```python
self.dicefocal_loss = DiceFocalLoss(
    gamma=focal_gamma, alpha=focal_alpha,
    lambda_dice=1.0, lambda_focal=1.0
)
```
- **Use Case**: Class imbalance, boundary refinement
- **Pros**: Focuses on hard examples, excellent for ET
- **Cons**: Requires gamma tuning

#### 4. Generalized Dice Loss
```python
self.generalized_dice_loss = GeneralizedDiceLoss(
    w_type='square'  # Automatic class reweighting
)
```
- **Use Case**: Severe class imbalance, no manual tuning
- **Pros**: Automatic inverse frequency weighting
- **Cons**: May over-emphasize rare classes

#### 5. Focal Loss
```python
self.focal_loss = FocalLoss(
    gamma=focal_gamma, alpha=focal_alpha
)
```
- **Use Case**: Hard example mining, component in hybrids
- **Pros**: Addresses class imbalance effectively
- **Cons**: Not recommended standalone for segmentation

#### 6. Tversky Loss
```python
self.tversky_loss = TverskyLoss(
    alpha=tversky_alpha, beta=tversky_beta
)
```
- **Use Case**: Precision/recall balance tuning
- **Pros**: Configurable FP/FN weighting
- **Cons**: Requires domain knowledge for parameter setting

#### 7. Hausdorff Distance Loss
```python
self.hausdorff_loss = HausdorffDTLoss(
    alpha=hausdorff_alpha, include_background=False
)
```
- **Use Case**: Boundary quality, shape preservation
- **Pros**: Excellent edge definition
- **Cons**: Computationally expensive, training instability

### Hybrid Loss Functions

#### 8. GDL + Focal + Tversky Hybrid
```python
def hybrid_gdl_focal_tversky_loss(self, outputs, labels):
    gdl_loss = self.generalized_dice_loss(outputs, labels)
    focal_loss = self.focal_loss(outputs, labels)
    tversky_loss = self.tversky_loss(outputs, labels)
    
    return (self.hparams.gdl_lambda * gdl_loss + 
            self.hparams.lambda_focal * focal_loss + 
            self.hparams.lambda_tversky * tversky_loss)
```
- **Use Case**: Competition settings, maximum performance
- **Pros**: Comprehensive optimization
- **Cons**: Complex parameter tuning

#### 9. Dice + Hausdorff Hybrid
```python
def hybrid_dice_hausdorff_loss(self, outputs, labels):
    dice_loss = self.dice_loss(outputs, labels)
    hausdorff_loss = self.hausdorff_loss(outputs, labels)
    
    return (self.hparams.lambda_dice * dice_loss + 
            self.hparams.lambda_hausdorff * hausdorff_loss)
```
- **Use Case**: Balance volume accuracy and boundary quality
- **Pros**: Medical imaging gold standard
- **Cons**: Hausdorff instability

### Adaptive Loss Functions

#### 10. Adaptive Structure-Boundary
**Strategy**: Dynamic scheduling from structure (Dice) to boundary (Focal) learning

```python
def _get_adaptive_weights(self):
    if self.adaptive_schedule_type == 'linear':
        progress = min(epoch / max_epoch, 1.0)
        dice_weight = max_weight * (1.0 - progress) + min_weight * progress
        focal_weight = min_weight * (1.0 - progress) + max_weight * progress
    # ... exponential and cosine variations
```

**Benefits:**
- **Early Training**: Structure learning with high Dice weight
- **Late Training**: Boundary refinement with high Focal weight
- **Smooth Transitions**: Configurable schedule types

#### 11. Adaptive Progressive Hybrid
**Strategy**: 3-phase progression adding complexity over time

```python
def _get_progressive_weights(self):
    if epoch < structure_epochs:
        return max_weight, min_weight, 0.0  # Dice only
    elif epoch < boundary_epochs:
        return max_weight * 0.7, max_weight * progress, 0.0  # Dice + Focal
    else:
        return max_weight * 0.7, max_weight * 0.8, min_weight + progress  # All
```

**Phases:**
1. **Structure** (0-30 epochs): Pure Dice
2. **Boundary** (30-50 epochs): Dice + Focal
3. **Details** (50+ epochs): Dice + Focal + Hausdorff

#### 12. Adaptive Complexity Cascade
**Strategy**: 4-stage cascading complexity introduction

```python
def _get_cascade_weights(self):
    progress = min(epoch / max_epoch, 1.0)
    
    if progress < 0.25:      # Stage 1: Pure Dice
        weights = {'dice': max_weight, 'dicece': 0.0, ...}
    elif progress < 0.5:     # Stage 2: Dice + DiceCE
        weights = {'dice': max_weight * 0.7, 'dicece': max_weight * progress, ...}
    # ... stages 3 and 4
```

**Stages:**
1. **Foundation** (0-25%): Dice only
2. **Classification** (25-50%): + DiceCE
3. **Hard Examples** (50-75%): + DiceFocal
4. **Comprehensive** (75-100%): All losses

#### 13. Adaptive Dynamic Hybrid
**Strategy**: Performance-based adaptation using validation metrics

```python
def _get_dynamic_weights(self):
    recent_dice = getattr(self, 'best_metric', 0.5)
    
    if recent_dice < 0.7:    # Poor performance - focus structure
        weights = {'gdl': max_weight, 'focal': min_weight, ...}
    elif recent_dice < 0.85: # Moderate - balanced approach
        weights = {'gdl': max_weight * 0.8, 'focal': max_weight * 0.6, ...}
    else:                    # Good - focus details
        weights = {'gdl': max_weight * 0.6, 'focal': max_weight, ...}
```

**Adaptation Logic:**
- **Poor Performance** (<0.7 Dice): Structure focus
- **Moderate Performance** (0.7-0.85): Balanced approach
- **Good Performance** (>0.85): Detail refinement

---

## Adaptive Scheduling

### Schedule Types

#### Linear Scheduling
```python
progress = min(epoch / max_epoch, 1.0)
weight = min_weight + (max_weight - min_weight) * progress
```
- **Use Case**: Smooth, predictable transitions
- **Pros**: Easy to understand and tune
- **Cons**: May be too gradual for some scenarios

#### Exponential Scheduling
```python
progress = min(epoch / max_epoch, 1.0)
weight = max_weight * (0.5 ** (progress * 3))
```
- **Use Case**: Rapid early transitions, gradual later
- **Pros**: Quick adaptation in early training
- **Cons**: May destabilize early training

#### Cosine Scheduling
```python
progress = min(epoch / max_epoch, 1.0)
weight = min_weight + (max_weight - min_weight) * 0.5 * (1 + cos(π * progress))
```
- **Use Case**: Smooth sinusoidal transitions
- **Pros**: Natural smooth transitions
- **Cons**: More complex behavior

### Configuration Parameters

```python
use_adaptive_scheduling=True    # Enable adaptive scheduling
adaptive_schedule_type='cosine' # Schedule type
structure_epochs=30            # Structure learning phase
boundary_epochs=50             # Boundary refinement phase
schedule_start_epoch=10        # When to start scheduling
min_loss_weight=0.1           # Minimum component weight
max_loss_weight=2.0           # Maximum component weight
```

---

## Training Pipeline

### Training Step Flow

```python
def training_step(self, batch, batch_idx):
    inputs, labels = batch["image"], batch["label"]
    
    # 1. Forward pass with optional modality attention
    outputs = self(inputs)
    
    # 2. Compute adaptive loss
    loss = self.compute_loss(outputs, labels)
    
    # 3. Apply post-processing
    outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
    
    # 4. Compute comprehensive metrics
    self.dice_metric(y_pred=outputs, y=labels)
    self.jaccard_metric(y_pred=outputs, y=labels)
    self.hausdorff_metric(y_pred=outputs, y=labels)
    
    # 5. Log all metrics
    train_dice = self.dice_metric.aggregate().item()
    precision, recall, f1 = self.compute_metrics(outputs, labels)
    
    return loss
```

### Validation Step Flow

```python
def validation_step(self, batch, batch_idx):
    val_inputs, val_labels = batch["image"], batch["label"]
    
    # 1. Sliding window inference
    val_outputs = sliding_window_inference(
        val_inputs, 
        roi_size=self.hparams.roi_size, 
        predictor=self.model, 
        overlap=self.overlap
    )
    
    # 2. Compute loss and metrics (same as training)
    val_loss = self.compute_loss(val_outputs, val_labels)
    
    # 3. Log sample images to WandB (first batch only)
    if batch_idx == 0:
        self.log_validation_images(val_inputs, val_outputs, val_labels)
    
    return {"val_loss": val_loss}
```

---

## Metrics and Logging

### Comprehensive Metrics

```python
# Segmentation metrics
self.dice_metric = DiceMetric(include_background=True, reduction="mean")
self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
self.jaccard_metric = MeanIoU(include_background=True, reduction="mean")
self.hausdorff_metric = HausdorffDistanceMetric(include_background=False)

# Custom precision, recall, F1
def compute_metrics(self, outputs, labels):
    tp = (outputs * labels).sum(dim=1)
    fp = (outputs * (1 - labels)).sum(dim=1)
    fn = ((1 - outputs) * labels).sum(dim=1)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision.mean(), recall.mean(), f1.mean()
```

### WandB Logging

```python
def log_validation_images(self, inputs, outputs, labels):
    if self.logger and hasattr(self.logger, "experiment"):
        # Extract middle slice for visualization
        img = inputs[0, 0].detach().cpu().numpy()
        pred = outputs[0][0].detach().cpu().numpy()
        label = labels[0, 0].detach().cpu().numpy()
        
        slice_idx = img.shape[-1] // 2
        
        self.logger.experiment.log({
            "val_image": wandb.Image(img[..., slice_idx]),
            "val_pred": wandb.Image(pred[..., slice_idx]),
            "val_label": wandb.Image(label[..., slice_idx]),
            "global_step": self.global_step
        })
```

### Best Model Tracking

```python
def on_validation_epoch_end(self):
    val_dice = self.dice_metric.aggregate().item()
    
    if val_dice > self.best_metric:
        self.best_metric = val_dice
        self.best_metric_epoch = self.current_epoch
        
        # Save best model
        torch.save(self.model.state_dict(), "best_metric_model_swinunetr_v2.pth")
        
        # Store all best metrics
        metric_batch = self.dice_metric_batch.aggregate()
        self.best_metric_tc = metric_batch[0].item()
        self.best_metric_wt = metric_batch[1].item()
        self.best_metric_et = metric_batch[2].item()
```

---

## Optimization

### Optimizer Configuration

```python
def configure_optimizers(self):
    optimizer = AdamW(
        self.model.parameters(), 
        lr=self.hparams.learning_rate, 
        weight_decay=self.hparams.weight_decay,
        betas=self.hparams.optimizer_betas,
        eps=self.hparams.optimizer_eps
    )
```

### Learning Rate Scheduling

#### Standard: Warmup + Cosine Annealing
```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### Advanced: Cosine Annealing with Warm Restarts
```python
if self.hparams.use_warm_restarts:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=self.hparams.restart_period,
        T_mult=self.hparams.restart_mult,
        eta_min=self.hparams.learning_rate * 0.01
    )
```

**Warm Restart Benefits:**
- **Local Minima Escape**: Periodic LR spikes help escape poor solutions
- **Multiple Attempts**: Each restart gives another chance to find better optima
- **Automatic**: No manual intervention required

### Post-Processing

```python
self.post_trans = Compose([
    Activations(sigmoid=True),           # Convert logits to probabilities
    AsDiscrete(threshold=0.5)            # Binarize predictions
])
```

**Configurable Threshold:**
- Default: 0.5 for balanced precision/recall
- Lower (0.3-0.4): Higher sensitivity, more false positives
- Higher (0.6-0.7): Higher specificity, fewer false positives

---

## Usage Examples

### Basic Training
```python
model = BrainTumorSegmentation(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_type='dicece',
    use_class_weights=True
)
```

### Advanced Training with Adaptive Losses
```python
model = BrainTumorSegmentation(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_type='adaptive_progressive_hybrid',
    use_adaptive_scheduling=True,
    adaptive_schedule_type='cosine',
    structure_epochs=40,
    boundary_epochs=70,
    use_warm_restarts=True,
    restart_period=30
)
```

### Competition Setup
```python
model = BrainTumorSegmentation(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_type='adaptive_dynamic_hybrid',
    use_adaptive_scheduling=True,
    use_warm_restarts=True,
    use_modality_attention=True,
    class_weights=(5.0, 1.0, 10.0),  # Aggressive ET weighting
    focal_gamma=2.8,
    tversky_alpha=0.25,
    threshold=0.4  # Higher sensitivity
)
```

---

## Key Implementation Details

### Thread Safety
- All metrics are reset after each epoch
- Validation metrics computed separately from training
- Proper tensor handling with `decollate_batch`

### Memory Optimization
- Sliding window inference with configurable overlap
- Gradient checkpointing in SwinUNETR backbone
- Mixed precision training (16-bit) enabled by default

### Reproducibility
- Fixed random seeds in data loading
- Deterministic operations where possible
- Comprehensive logging for experiment tracking

### Error Handling
- Robust Hausdorff distance computation with NaN handling
- Graceful degradation when WandB logging fails
- Automatic metric resets to prevent accumulation errors

This pipeline implementation represents a state-of-the-art approach to medical image segmentation, combining proven architectures with novel adaptive optimization strategies specifically designed for the challenges of brain tumor segmentation.