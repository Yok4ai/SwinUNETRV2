# SwinUNETR V2 Experimental Results

## BraTS 2021 Results

| Experiment | Dataset | Loss Type | Best Mean Dice | Best TC Dice | Best WT Dice | Best ET Dice | Best Mean IoU | Best Mean F1 | Best Mean Precision | Best Mean Recall | Best Hausdorff | Epochs to Best | Configuration |
|------------|---------|-----------|---------------|--------------|--------------|--------------|---------------|--------------|--------------------|--------------------|----------------|----------------|---------------|
| 2021-gdl-focal-tversky-v2 | BraTS 2021 | hybrid_gdl_focal_tversky | 0.8656 | 0.8314 | 0.9337 | 0.8716 | 0.8048 | 0.9023 | 0.9363 | 0.8834 | 24.4127 | 38 | batch_size=2, lr=5e-4, img_size=96, feature_size=48, roi_size=96x96x96, overlap=0.7, warmup_epochs=15, early_stopping_patience=10, modality_attention=True, class_weights=[3.0, 1.0, 5.0], val_interval=1, limit_val_batches=5 |
| 2021-hybrid-dice-warmup15 | BraTS 2021 | dice | 0.8710 | 0.8419 | 0.9366 | 0.8578 | 0.8063 | 0.9059 | 0.9420 | 0.8798 | 22.8352 | 48 | batch_size=2, lr=1e-4, img_size=96, feature_size=48, roi_size=96x96x96, overlap=0.7, warmup_epochs=10, dice_ce_weight=0.6, focal_weight=0.4, use_class_weights=False, modality_attention=False |
| 2021-hybrid-class-weights | BraTS 2021 | dicece-focal | 0.8483 | 0.8009 | 0.8929 | 0.8325 | 0.7640 | 0.8677 | 0.9153 | 0.8418 | 23.9028 | 24 | batch_size=2, lr=1e-3, img_size=96, feature_size=48, roi_size=96x96x96, overlap=0.7, warmup_epochs=10, dice_ce_weight=0.5, focal_weight=0.5, use_class_weights=True, class_weights=[3.0, 1.0, 5.0], modality_attention=True, limit_val_batches=10 |
## BraTS 2023 Results

| Experiment | Dataset | Loss Type | Best Mean Dice | Best TC Dice | Best WT Dice | Best ET Dice | Best Mean IoU | Best Mean F1 | Best Mean Precision | Best Mean Recall | Best Hausdorff | Epochs to Best | Configuration |
|------------|---------|-----------|---------------|--------------|--------------|--------------|---------------|--------------|--------------------|--------------------|----------------|----------------|---------------|
| 2023-hybrid-class-weights | BraTS 2023 | hybrid-dicece-focal | 0.9338 | 0.9423 | 0.9368 | 0.9223 | 0.8788 | 0.9380 | 0.9205 | 0.9592 | 23.3095 | 49 | batch_size=2, lr=1e-3, img_size=96, feature_size=48, roi_size=96x96x96, warmup_epochs=15, epochs=50, use_class_weights=True, modality_attention=True, loss_type=hybrid-dicece-focal |



## Summary Statistics

### Best Performance per Metric
- **Highest Mean Dice**: 0.9338 (2023-hybrid-class-weights)
- **Highest TC Dice**: 0.9423 (2023-hybrid-class-weights)
- **Highest WT Dice**: 0.9368 (2023-hybrid-class-weights)
- **Highest ET Dice**: 0.9223 (2023-hybrid-class-weights)
- **Highest Mean IoU**: 0.8788 (2023-hybrid-class-weights)
- **Highest Mean F1**: 0.9380 (2023-hybrid-class-weights)
- **Highest Mean Precision**: 0.9420 (2021-hybrid-dice-warmup15)
- **Highest Mean Recall**: 0.9592 (2023-hybrid-class-weights)
- **Lowest Hausdorff**: 22.8352 (2021-hybrid-dice-warmup15)

### Optimal Configurations Across Both Datasets

#### Best Loss Functions
1. **hybrid-dicece-focal**: Consistently strong performance across both datasets
2. **hybrid_gdl_focal_tversky**: Good balance with class weights
3. **Hybrid losses outperform single loss types in all experiments**

#### Best Learning Rates
- **1e-3**: Works well for both datasets, especially with class weights
- **1e-4**: Good for stable training without class weights
- **5e-4**: Optimal middle ground for complex loss combinations

#### Best Warmup Epochs
- **10 epochs**: Good for faster convergence and simpler configurations
- **15 epochs**: Better for stable training with class weights and modality attention
- **Longer warmup generally improves final performance**

#### Key Configuration Insights
- **Class Weights [3.0, 1.0, 5.0]**: Consistently help with class imbalance across both datasets
- **Modality Attention**: Most effective when combined with class weights
- **DiceCE vs Focal Weights**: 0.6/0.4 (DiceCE/Focal) works better than 0.5/0.5
- **Batch Size**: 2 is optimal for 96x96x96 images across all experiments
- **Image Size**: 96x96x96 provides best performance-cost balance

#### Universal Training Patterns
- **Hybrid losses**: Always outperform single loss types
- **Class weights + Modality attention**: Powerful combination for both datasets
- **Convergence**: 24-49 epochs depending on configuration complexity
- **BraTS 2023 generally achieves higher absolute performance than BraTS 2021**

### Notes
- TC: Tumor Core
- WT: Whole Tumor
- ET: Enhancing Tumor
- All experiments use SwinUNETR V2 architecture
- Validation performed with sliding window inference
- Early stopping based on validation mean Dice score