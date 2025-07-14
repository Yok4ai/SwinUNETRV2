# run.py
import sys
import os
from kaggle_setup import setup_kaggle_notebook
from main import main
import argparse
import torch
import warnings

# CLI argument parsing for key parameters
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='brats2023', choices=['brats2021', 'brats2023'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of data loader workers')
    parser.add_argument('--img_size', type=int, default=96, help='Input image size')
    parser.add_argument('--feature_size', type=int, default=48, help='Model feature size')
    parser.add_argument('--loss_type', type=str, default='dice', 
                        choices=['dice', 'dicece', 'dicefocal', 'generalized_dice', 'generalized_dice_focal', 
                                'focal', 'tversky', 'hausdorff', 'hybrid_gdl_focal_tversky', 'hybrid_dice_hausdorff',
                                'adaptive_structure_boundary', 'adaptive_progressive_hybrid', 
                                'adaptive_complexity_cascade', 'adaptive_dynamic_hybrid'], 
                        help='Loss function type')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for LR scheduler')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for loss (default: False)')
    parser.add_argument('--use_modality_attention', action='store_true', help='Enable Modality Attention module (default: False)')
    parser.add_argument('--overlap', type=float, default=0.7, help='Sliding window inference overlap (default: 0.7)')
    parser.add_argument('--class_weights', type=float, nargs=3, default=[3.0, 1.0, 5.0], help='Class weights for TC, WT, ET (default: 3.0 1.0 5.0)')
    # Removed dice_ce_weight and focal_weight - use lambda parameters instead
    # New loss function parameters
    parser.add_argument('--tversky_alpha', type=float, default=0.5, help='Tversky loss alpha parameter (default: 0.5)')
    parser.add_argument('--tversky_beta', type=float, default=0.5, help='Tversky loss beta parameter (default: 0.5)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=None, help='Focal loss alpha parameter (default: None)')
    parser.add_argument('--gdl_weight_type', type=str, default='square', choices=['square', 'simple', 'uniform'], 
                        help='Generalized Dice Loss weight type (default: square)')
    parser.add_argument('--gdl_lambda', type=float, default=1.0, help='Generalized Dice Loss lambda parameter (default: 1.0)')
    parser.add_argument('--hausdorff_alpha', type=float, default=2.0, help='Hausdorff loss alpha parameter (default: 2.0)')
    parser.add_argument('--lambda_dice', type=float, default=1.0, help='Lambda weight for Dice loss component (default: 1.0)')
    parser.add_argument('--lambda_focal', type=float, default=1.0, help='Lambda weight for Focal loss component (default: 1.0)')
    parser.add_argument('--lambda_tversky', type=float, default=1.0, help='Lambda weight for Tversky loss component (default: 1.0)')
    parser.add_argument('--lambda_hausdorff', type=float, default=1.0, help='Lambda weight for Hausdorff loss component (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for post-processing discrete output (default: 0.5)')
    parser.add_argument('--roi_size', type=int, nargs=3, default=[96, 96, 96], help='ROI size for sliding window inference (default: 96 96 96)')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience epochs (default: 15)')
    parser.add_argument('--limit_val_batches', type=int, default=5, help='Limit validation batches for faster validation (default: 5)')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval in epochs (default: 1)')
    # Adaptive loss scheduling parameters
    parser.add_argument('--use_adaptive_scheduling', action='store_true', help='Enable adaptive loss scheduling (default: False)')
    parser.add_argument('--adaptive_schedule_type', type=str, default='linear', choices=['linear', 'exponential', 'cosine'],
                        help='Type of adaptive scheduling (default: linear)')
    parser.add_argument('--structure_epochs', type=int, default=30, help='Epochs to focus on structure learning (default: 30)')
    parser.add_argument('--boundary_epochs', type=int, default=50, help='Epochs to focus on boundary refinement (default: 50)')
    parser.add_argument('--schedule_start_epoch', type=int, default=10, help='Epoch to start adaptive scheduling (default: 10)')
    parser.add_argument('--min_loss_weight', type=float, default=0.1, help='Minimum weight for any loss component (default: 0.1)')
    parser.add_argument('--max_loss_weight', type=float, default=2.0, help='Maximum weight for any loss component (default: 2.0)')
    # Warm restart parameters for local minima escape
    parser.add_argument('--use_warm_restarts', action='store_true', help='Enable cosine annealing with warm restarts (default: False)')
    parser.add_argument('--restart_period', type=int, default=20, help='Restart period in epochs (default: 20)')
    parser.add_argument('--restart_mult', type=int, default=1, help='Restart period multiplier (default: 1)')
    return parser.parse_args()

cli_args = parse_cli_args()

"""
## Example usage - Multiple loss function options:

# Standard Dice Loss
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type dice \
  --learning_rate 5e-4 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# DiceFocal Loss
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type dicefocal \
  --learning_rate 5e-4 \
  --focal_gamma 2.0 \
  --lambda_dice 1.0 \
  --lambda_focal 1.0 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Hybrid: Generalized Dice + Focal + Tversky
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type hybrid_gdl_focal_tversky \
  --learning_rate 5e-4 \
  --gdl_lambda 1.0 \
  --lambda_focal 0.5 \
  --lambda_tversky 0.3 \
  --focal_gamma 2.0 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Hybrid: Dice + Hausdorff
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 50 \
  --batch_size 1 \
  --loss_type hybrid_dice_hausdorff \
  --learning_rate 5e-4 \
  --lambda_dice 1.0 \
  --lambda_hausdorff 0.1 \
  --hausdorff_alpha 2.0 \
  --use_class_weights \
  --class_weights 4.0 1.0 6.0

# Adaptive Structure-Boundary Scheduling
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type adaptive_structure_boundary \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --adaptive_schedule_type cosine \
  --schedule_start_epoch 15 \
  --min_loss_weight 0.2 \
  --max_loss_weight 1.5 \
  --use_class_weights

# Adaptive Progressive Hybrid
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --loss_type adaptive_progressive_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --schedule_start_epoch 10 \
  --use_class_weights

# Adaptive Complexity Cascade
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type adaptive_complexity_cascade \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --adaptive_schedule_type linear \
  --use_class_weights

# Adaptive Dynamic Hybrid (Performance-based)
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type adaptive_dynamic_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --schedule_start_epoch 20 \
  --use_class_weights

# Warm Restarts for Local Minima Escape
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 1 \
  --loss_type generalized_dice_focal \
  --learning_rate 1e-4 \
  --use_warm_restarts \
  --restart_period 25 \
  --restart_mult 1 \
  --use_class_weights

# Adaptive + Warm Restarts (SOTA Combination)
!python /kaggle/working/SwinUNETRV2/kaggle_run.py \
  --dataset brats2023 \
  --epochs 120 \
  --batch_size 1 \
  --loss_type adaptive_progressive_hybrid \
  --learning_rate 1e-4 \
  --use_adaptive_scheduling \
  --structure_epochs 40 \
  --boundary_epochs 70 \
  --use_warm_restarts \
  --restart_period 30 \
  --use_class_weights

# Alternative: Memory-optimized for smaller GPU
!python kaggle_run.py \
  --dataset brats2023 \
  --epochs 75 \
  --batch_size 1 \
  --img_size 96 \
  --roi_size 96 96 96 \
  --learning_rate 3e-4 \
  --warmup_epochs 8 \
  --early_stopping_patience 12 \
  --limit_val_batches 5 \
  --use_class_weights \

# Conservative: Standard settings for longer training
!python kaggle_run.py \
  --dataset brats2023 \
  --epochs 100 \
  --batch_size 2 \
  --img_size 96 \
  --roi_size 96 96 96 \
  --learning_rate 1e-4 \
  --warmup_epochs 10 \
  --early_stopping_patience 15 \
  --use_class_weights

# Omit --use_class_weights, --use_modality_attention to disable them (they are store_true flags)
"""

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook(cli_args.dataset)
print(f"Dataset prepared in: {output_dir}")

# Experimental Configuration
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',
    batch_size=cli_args.batch_size,
    num_workers=cli_args.num_workers,
    pin_memory=True,
    persistent_workers=False,
    dataset=cli_args.dataset,  # Use the selected dataset
    
    # Model parameters
    img_size=cli_args.img_size,
    in_channels=4,
    out_channels=3,
    feature_size=cli_args.feature_size,

    # Training parameters
    learning_rate=cli_args.learning_rate,  # Now from CLI
    weight_decay=1e-5,
    warmup_epochs=cli_args.warmup_epochs,
    epochs=cli_args.epochs,
    accelerator='gpu',
    devices='auto',
    precision='16-mixed',
    strategy="ddp",
    log_every_n_steps=1,
    enable_checkpointing=True,
    benchmark=True,
    profiler="simple",
    use_amp=True,  # Enable mixed precision
    gradient_clip_val=1.0,
    use_v2=True,
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    downsample="mergingv2",
    
    # Enhanced model options
    use_class_weights=cli_args.use_class_weights,
    use_modality_attention=cli_args.use_modality_attention,
    
    # Loss and training configuration
    class_weights=tuple(cli_args.class_weights),  # TC, WT, ET
    threshold=cli_args.threshold,
    optimizer_betas=(0.9, 0.999),
    optimizer_eps=1e-8,
    
    # Validation settings
    val_interval=cli_args.val_interval,
    save_interval=1,
    early_stopping_patience=cli_args.early_stopping_patience,
    limit_val_batches=cli_args.limit_val_batches,
    
    # Inference parameters
    roi_size=cli_args.roi_size,
    sw_batch_size=1,
    overlap=cli_args.overlap,
    loss_type=cli_args.loss_type,
    
    # Loss function parameters
    tversky_alpha=cli_args.tversky_alpha,
    tversky_beta=cli_args.tversky_beta,
    focal_gamma=cli_args.focal_gamma,
    focal_alpha=cli_args.focal_alpha,
    gdl_weight_type=cli_args.gdl_weight_type,
    gdl_lambda=cli_args.gdl_lambda,
    hausdorff_alpha=cli_args.hausdorff_alpha,
    lambda_dice=cli_args.lambda_dice,
    lambda_focal=cli_args.lambda_focal,
    lambda_tversky=cli_args.lambda_tversky,
    lambda_hausdorff=cli_args.lambda_hausdorff,
    
    # Adaptive loss scheduling parameters
    use_adaptive_scheduling=cli_args.use_adaptive_scheduling,
    adaptive_schedule_type=cli_args.adaptive_schedule_type,
    structure_epochs=cli_args.structure_epochs,
    boundary_epochs=cli_args.boundary_epochs,
    schedule_start_epoch=cli_args.schedule_start_epoch,
    min_loss_weight=cli_args.min_loss_weight,
    max_loss_weight=cli_args.max_loss_weight,
    
    # Warm restart parameters
    use_warm_restarts=cli_args.use_warm_restarts,
    restart_period=cli_args.restart_period,
    restart_mult=cli_args.restart_mult,
)

# Print final configuration summary
print("\n=== 🚀 ADAPTIVE SWINUNETR CONFIGURATION ===")
print(f"🎯 Batch size: {args.batch_size}")
print(f"📐 Image size: {args.img_size}")
print(f"⚡ Learning rate: {args.learning_rate}")
print(f"🔥 Warmup epochs: {args.warmup_epochs}")
print(f"🧮 Loss type: {args.loss_type}")
print(f"🔄 SW batch size: {args.sw_batch_size}")
print(f"📊 Total epochs: {args.epochs}")
print(f"🗂️ Dataset: {args.dataset}")
print(f"🏋️ Use class weights: {args.use_class_weights}")
print(f"🧠 Use modality attention: {args.use_modality_attention}")
print(f"⚖️ Class weights: {args.class_weights}")
# Removed dice_ce_weight and focal_weight - using lambda parameters instead
print(f"🎛️ Tversky α: {args.tversky_alpha}, β: {args.tversky_beta}")
print(f"🎯 Focal γ: {args.focal_gamma}, α: {args.focal_alpha}")
print(f"🏗️ GDL weight type: {args.gdl_weight_type}, λ: {args.gdl_lambda}")
print(f"📏 Hausdorff α: {args.hausdorff_alpha}")
print(f"⚖️ Loss weights - Dice: {args.lambda_dice}, Focal: {args.lambda_focal}, Tversky: {args.lambda_tversky}, Hausdorff: {args.lambda_hausdorff}")
print(f"🎚️ Threshold: {args.threshold}")
print(f"🔲 ROI size: {args.roi_size}")
print(f"⏹️ Early stop patience: {args.early_stopping_patience}")
print(f"🔢 Limit val batches: {args.limit_val_batches}")
print(f"📅 Val interval: {args.val_interval}")
print(f"🔄 Adaptive scheduling: {args.use_adaptive_scheduling}")
if args.use_adaptive_scheduling:
    print(f"📈 Schedule type: {args.adaptive_schedule_type}")
    print(f"🏗️ Structure epochs: {args.structure_epochs}")
    print(f"🎯 Boundary epochs: {args.boundary_epochs}")
    print(f"🚀 Schedule start: {args.schedule_start_epoch}")
    print(f"⚖️ Weight range: {args.min_loss_weight} - {args.max_loss_weight}")
print(f"🔥 Warm restarts: {args.use_warm_restarts}")
if args.use_warm_restarts:
    print(f"🔄 Restart period: {args.restart_period} epochs")
    print(f"📊 Restart multiplier: {args.restart_mult}")

def run_with_error_handling():
    """Run training with comprehensive error handling"""
    try:
        main(args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
        else:
            print(f"❌ Runtime error: {e}")
        raise e
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise e
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise e

# Start optimized training
if __name__ == "__main__":
    run_with_error_handling()