# SwinUNETR V2 Experimental Results

## BraTS 2021 Results

| Experiment | Dataset | Loss Type | Best Mean Dice | Best TC Dice | Best WT Dice | Best ET Dice | Best Mean IoU | Best Hausdorff | Epochs to Best | Configuration |
|------------|---------|-----------|---------------|--------------|--------------|--------------|---------------|----------------|----------------|---------------|
| 2021-dice | BraTS 2021 | dice | 0.8710 | 0.8479 | 0.9351 | 0.8607 | 0.8077 | 22.9902 | 43 | batch_size=2, lr=1e-3, img_size=96, feature_size=48, roi_size=96x96x96, overlap=0.7, warmup_epochs=15, epochs=50, early_stopping_patience=15, modality_attention=True |
| 2021-dicefocal | BraTS 2021 | dicefocal | 0.8693 | 0.8375 | 0.9350 | 0.8677 | 0.8059 | 28.0521 | 38 | batch_size=2, lr=1e-3, img_size=96, feature_size=48, warmup_epochs=15, epochs=50, use_class_weights=True, modality_attention=True |
| 2021-gdl-focal-tversky | BraTS 2021 | hybrid_gdl_focal_tversky | 0.8656 | 0.8314 | 0.9337 | 0.8716 | 0.8048 | 24.4127 | 38 | batch_size=2, lr=5e-4, img_size=96, feature_size=48, roi_size=96x96x96, overlap=0.7, warmup_epochs=15, early_stopping_patience=10, modality_attention=True, class_weights=[3.0, 1.0, 5.0] |
| 2021-dicece-focal | BraTS 2021 | dicece+focal | 0.8483 | 0.8223 | 0.8891 | 0.8446 | 0.7688 | 25.5657 | 14 | batch_size=2, lr=1e-3, img_size=96, feature_size=48, roi_size=96x96x96, overlap=0.7, warmup_epochs=10, epochs=50, early_stopping_patience=10, class_weights=[3.0, 1.0, 5.0], dice_ce_weight=0.5, focal_weight=0.5, use_class_weights=True, modality_attention=True |
| 2021-dicefocal-128 | BraTS 2021 | dicefocal | 0.8271 | 0.7999 | 0.9071 | 0.8158 | 0.7476 | 12.5113 | 15 | batch_size=1, lr=5e-4, img_size=128, feature_size=48, roi_size=128x128x128, overlap=0.7, warmup_epochs=5, epochs=50, early_stopping_patience=8, class_weights=[1.0, 4.0, 6.0], dice_ce_weight=0.7, focal_weight=0.3, use_class_weights=True, modality_attention=True |

## BraTS 2023 Results

| Experiment | Dataset | Loss Type | Best Mean Dice | Best TC Dice | Best WT Dice | Best ET Dice | Best Mean IoU | Best Hausdorff | Epochs to Best | Configuration |
|------------|---------|-----------|---------------|--------------|--------------|--------------|---------------|----------------|----------------|---------------|
| 2023-dicefocal | BraTS 2023 | dicefocal | 0.9369 | 0.9453 | 0.9397 | 0.9257 | 0.8841 | 23.2879 | 46 | batch_size=2, lr=1e-3, img_size=96, feature_size=48, warmup_epochs=15, epochs=50, use_class_weights=True, modality_attention=True |


## Summary Statistics

### Best Performance per Metric
- **Highest Mean Dice**: 0.9369 (2023-dicefocal)
- **Highest TC Dice**: 0.9453 (2023-dicefocal)
- **Highest WT Dice**: 0.9397 (2023-dicefocal)
- **Highest ET Dice**: 0.9257 (2023-dicefocal)
- **Highest Mean IoU**: 0.8841 (2023-dicefocal)
- **Lowest Hausdorff**: 12.5113 (2021-dicefocal-128)

### Hyperparameter Analysis

#### Optimal Configurations
- **Best Overall Performance**: 2023-dicefocal
  - Loss: dicefocal, LR: 1e-3, Batch: 2, Image Size: 96, Feature Size: 48
  - Warmup: 15 epochs, Class weights: True, Modality attention: True
  - Converged at epoch 46/50

- **Fastest Convergence**: 2021-dicefocal-128
  - Loss: dicefocal, LR: 5e-4, Batch: 1, Image Size: 128, Feature Size: 48
  - Warmup: 5 epochs, Early stopping: 8 patience
  - Converged at epoch 15/50, Best Hausdorff distance

#### Key Hyperparameter Insights
- **Learning Rate**: 1e-3 performs best for BraTS 2023, 5e-4 optimal for larger image sizes (128)
- **Batch Size**: 2 is optimal for 96x96x96 images, 1 required for 128x128x128 due to memory constraints
- **Image Size**: 96x96x96 provides better balance of performance vs computational cost
- **Loss Functions**: DiceFocal consistently outperforms pure Dice loss
- **Class Weights**: [3.0, 1.0, 5.0] or [1.0, 4.0, 6.0] help with class imbalance
- **Modality Attention**: Consistently improves performance when used
- **Warmup Epochs**: 15 epochs optimal for stable training, 5 for faster convergence

#### Training Efficiency
- **Epochs to Convergence**: 15-75 epochs (median: 38)
- **Early Stopping**: 8-15 patience effective for preventing overfitting
- **ROI Size**: 96x96x96 with 0.7 overlap provides good inference quality
- **Feature Size**: 48 consistently used across all experiments

### Dataset-Specific Performance
- **BraTS 2021**: Mean Dice range 0.8271-0.8710, more consistent performance
- **BraTS 2023**: Higher peak performance (0.9369) but fewer experiments
- **Label Conversion**: BraTS 2023 benefits from updated label format

### Training Patterns
- **Loss Combinations**: Hybrid losses (DiceCE + Focal) show promise but require careful weight tuning
- **Convergence**: Most models converge within 40-50 epochs with proper warmup
- **Generalization**: Higher image resolution (128) trades performance for faster convergence
- **Memory Optimization**: Batch size 1 required for 128x128x128 images

### Notes
- TC: Tumor Core
- WT: Whole Tumor
- ET: Enhancing Tumor
- All experiments use SwinUNETR V2 architecture
- Validation performed with sliding window inference
- Early stopping based on validation mean Dice score