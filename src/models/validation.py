import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss, GeneralizedDiceLoss, GeneralizedDiceFocalLoss, TverskyLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete, RandFlipd, RandRotate90d, Lambda, ToTensord
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms.spatial.array import Flip, Rotate90
from monai.transforms.utils import map_classes_to_indices
import json
import argparse
from pathlib import Path
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import existing components
from .swinunetrplus import SwinUNETR
from ..data.augmentations import get_transforms
from ..data.convert_labels import ConvertLabels

class TTATransforms:
    """Test Time Augmentation transforms for medical image segmentation."""
    
    def __init__(self, spatial_axes: List[int] = [0, 1, 2]):
        self.spatial_axes = spatial_axes
        
    def get_tta_transforms(self) -> List[Dict[str, Any]]:
        """
        Get list of TTA transforms for medical image segmentation.
        Each transform is a dict with 'forward' and 'inverse' functions.
        """
        transforms = []
        
        # Original (no augmentation)
        transforms.append({
            'name': 'original',
            'forward': lambda x: x,
            'inverse': lambda x: x
        })
        
        # Horizontal flip (axis 0)
        transforms.append({
            'name': 'flip_0',
            'forward': lambda x: torch.flip(x, dims=[2]),  # D dimension
            'inverse': lambda x: torch.flip(x, dims=[2])
        })
        
        # Vertical flip (axis 1) 
        transforms.append({
            'name': 'flip_1',
            'forward': lambda x: torch.flip(x, dims=[3]),  # H dimension
            'inverse': lambda x: torch.flip(x, dims=[3])
        })
        
        # Depth flip (axis 2)
        transforms.append({
            'name': 'flip_2', 
            'forward': lambda x: torch.flip(x, dims=[4]),  # W dimension
            'inverse': lambda x: torch.flip(x, dims=[4])
        })
        
        # 90-degree rotation in axial plane
        transforms.append({
            'name': 'rot90_01',
            'forward': lambda x: torch.rot90(x, k=1, dims=[3, 4]),  # H, W
            'inverse': lambda x: torch.rot90(x, k=-1, dims=[3, 4])
        })
        
        # 180-degree rotation in axial plane
        transforms.append({
            'name': 'rot180_01',
            'forward': lambda x: torch.rot90(x, k=2, dims=[3, 4]),  # H, W
            'inverse': lambda x: torch.rot90(x, k=-2, dims=[3, 4])
        })
        
        # 270-degree rotation in axial plane
        transforms.append({
            'name': 'rot270_01',
            'forward': lambda x: torch.rot90(x, k=3, dims=[3, 4]),  # H, W
            'inverse': lambda x: torch.rot90(x, k=-3, dims=[3, 4])
        })
        
        # Combined transforms
        transforms.append({
            'name': 'flip_0_rot90',
            'forward': lambda x: torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[3, 4]),
            'inverse': lambda x: torch.flip(torch.rot90(x, k=-1, dims=[3, 4]), dims=[2])
        })
        
        return transforms

class StandaloneValidationPipeline:
    """Standalone validation pipeline with Test Time Augmentation support."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 data_dir: str,
                 dataset: str = "brats2023",
                 batch_size: int = 1,
                 num_workers: int = 4,
                 roi_size: Tuple[int, int, int] = (96, 96, 96),
                 sw_batch_size: int = 1,
                 overlap: float = 0.7,
                 threshold: float = 0.5,
                 use_tta: bool = True,
                 tta_merge_mode: str = "mean",  # "mean", "median", "max"
                 device: str = "cuda",
                 class_weights: Tuple[float, float, float] = (3.0, 1.0, 5.0),
                 # Model architecture parameters
                 feature_size: int = 48,
                 use_v2: bool = True,
                 depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
                 num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
                 downsample: str = "mergingv2",
                 use_modality_attention: bool = False,
                 # Validation settings
                 max_batches: int = None,  # Limit number of validation batches
                 save_predictions: bool = False,
                 output_dir: str = "./validation_results",
                 log_to_wandb: bool = False,
                 wandb_project: str = "swinunetr-validation"):
        
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.threshold = threshold
        self.use_tta = use_tta
        self.tta_merge_mode = tta_merge_mode
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_weights = class_weights
        
        # Model parameters
        self.feature_size = feature_size
        self.use_v2 = use_v2
        self.depths = depths
        self.num_heads = num_heads
        self.downsample = downsample
        self.use_modality_attention = use_modality_attention
        
        # Validation settings
        self.max_batches = max_batches
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir)
        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        
        # Initialize components
        self.model = None
        self.val_loader = None
        self.tta_transforms = TTATransforms() if use_tta else None
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.jaccard_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
        
        # Post-processing
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=self.threshold)])
        
        # Results storage
        self.results = {
            'mean_dice': [],
            'dice_tc': [],
            'dice_wt': [], 
            'dice_et': [],
            'mean_iou': [],
            'hausdorff': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def setup(self):
        """Setup model, data loader, and logging."""
        print("Setting up validation pipeline...")
        
        # Create output directory
        if self.save_predictions:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize WandB if requested
        if self.log_to_wandb:
            wandb.init(
                project=self.wandb_project,
                name=f"validation_{self.dataset}_{int(time.time())}",
                config={
                    "checkpoint_path": self.checkpoint_path,
                    "dataset": self.dataset,
                    "use_tta": self.use_tta,
                    "tta_merge_mode": self.tta_merge_mode,
                    "roi_size": self.roi_size,
                    "overlap": self.overlap,
                    "threshold": self.threshold
                }
            )
        
        # Setup model
        self._setup_model()
        
        # Setup data loader
        self._setup_dataloader()
        
        print(f"Model loaded from: {self.checkpoint_path}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Total batches: {len(self.val_loader)}")
        if self.max_batches:
            print(f"Limited to: {self.max_batches} batches")
        print(f"Using TTA: {self.use_tta}")
        if self.use_tta:
            print(f"TTA merge mode: {self.tta_merge_mode}")
            print(f"TTA transforms: {len(self.tta_transforms.get_tta_transforms())}")
        
    def _setup_model(self):
        """Initialize and load the SwinUNETR model."""
        # Create model
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=self.feature_size,
            use_checkpoint=True,
            use_v2=self.use_v2,
            spatial_dims=3,
            depths=self.depths,
            num_heads=self.num_heads,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            downsample=self.downsample,
        )
        
        # Load checkpoint
        if self.checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint - extract only the SwinUNETR model weights
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    # Remove the 'model.' prefix to match direct SwinUNETR
                    new_key = key[6:]
                    model_state_dict[new_key] = value
            self.model.load_state_dict(model_state_dict, strict=False)
        else:
            # PyTorch state dict
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.to(self.device)
        self.model.eval()
        
    def _setup_dataloader(self):
        """Setup validation data loader."""
        # Get transforms
        _, val_transforms = get_transforms(img_size=self.roi_size[0], dataset=self.dataset)
        
        # Load dataset
        dataset_path = self.data_dir if self.data_dir.endswith('dataset.json') else os.path.join(self.data_dir, "dataset.json")
        with open(dataset_path) as f:
            datalist = json.load(f)["training"]
        
        # Use validation split (20% of data)
        _, val_files = train_test_split(datalist, test_size=0.2, random_state=42)
        
        # Create validation dataset and loader
        val_ds = Dataset(data=val_files, transform=val_transforms)
        self.val_loader = DataLoader(
            val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def predict_with_tta(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform prediction with Test Time Augmentation.
        
        Args:
            inputs: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Ensemble prediction tensor
        """
        if not self.use_tta:
            return sliding_window_inference(
                inputs, 
                roi_size=self.roi_size, 
                sw_batch_size=self.sw_batch_size,
                predictor=self.model, 
                overlap=self.overlap
            )
        
        tta_transforms = self.tta_transforms.get_tta_transforms()
        predictions = []
        
        for transform_dict in tta_transforms:
            # Apply forward transform
            augmented_inputs = transform_dict['forward'](inputs)
            
            # Predict
            with torch.no_grad():
                pred = sliding_window_inference(
                    augmented_inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=self.overlap
                )
            
            # Apply inverse transform to prediction
            pred = transform_dict['inverse'](pred)
            predictions.append(pred)
        
        # Ensemble predictions
        predictions = torch.stack(predictions, dim=0)  # (n_transforms, B, C, D, H, W)
        
        if self.tta_merge_mode == "mean":
            ensemble_pred = torch.mean(predictions, dim=0)
        elif self.tta_merge_mode == "median":
            ensemble_pred = torch.median(predictions, dim=0)[0]
        elif self.tta_merge_mode == "max":
            ensemble_pred = torch.max(predictions, dim=0)[0]
        else:
            raise ValueError(f"Unknown TTA merge mode: {self.tta_merge_mode}")
            
        return ensemble_pred
        
    def compute_metrics(self, outputs: List[torch.Tensor], labels: List[torch.Tensor]) -> Dict[str, float]:
        """Compute comprehensive metrics for the batch."""
        # Stack outputs and labels if they're lists
        outputs = torch.stack(outputs) if isinstance(outputs, list) else outputs
        labels = torch.stack(labels) if isinstance(labels, list) else labels
        
        outputs = outputs.float()
        labels = labels.float()
        
        # Flatten all but batch dimension for precision/recall/F1
        outputs_flat = outputs.view(outputs.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        
        # True Positives, False Positives, False Negatives
        tp = (outputs_flat * labels_flat).sum(dim=1)
        fp = (outputs_flat * (1 - labels_flat)).sum(dim=1)
        fn = ((1 - outputs_flat) * labels_flat).sum(dim=1)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item()
        }
        
    def validate(self) -> Dict[str, float]:
        """Run validation with TTA and return comprehensive metrics."""
        print("\nStarting validation...")
        
        self.model.eval()
        
        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()
        self.jaccard_metric.reset()
        self.hausdorff_metric.reset()
        
        total_time = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Break if we've reached the maximum number of batches
                if self.max_batches and batch_idx >= self.max_batches:
                    print(f"\nReached maximum batch limit ({self.max_batches}), stopping validation.")
                    break
                    
                start_time = time.time()
                
                val_inputs = batch["image"].to(self.device)
                val_labels = batch["label"].to(self.device)
                
                # Predict with or without TTA
                val_outputs = self.predict_with_tta(val_inputs)
                
                # Post-process outputs
                val_outputs_processed = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                val_labels_processed = decollate_batch(val_labels)
                
                # Compute metrics
                self.dice_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                self.dice_metric_batch(y_pred=val_outputs_processed, y=val_labels_processed)
                self.jaccard_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                self.hausdorff_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                
                # Compute additional metrics
                additional_metrics = self.compute_metrics(val_outputs_processed, val_labels_processed)
                
                # Store batch results
                mean_dice = self.dice_metric.aggregate().item()
                mean_iou = self.jaccard_metric.aggregate().item()
                dice_batch = self.dice_metric_batch.aggregate()
                
                hausdorff_values = self.hausdorff_metric.aggregate(reduction='none')
                if not isinstance(hausdorff_values, torch.Tensor):
                    hausdorff_values = torch.tensor(hausdorff_values)
                valid = torch.isfinite(hausdorff_values)
                hausdorff = hausdorff_values[valid].mean().item() if valid.any() else float('nan')
                
                # Store results
                self.results['mean_dice'].append(mean_dice)
                self.results['dice_tc'].append(dice_batch[0].item())
                self.results['dice_wt'].append(dice_batch[1].item())
                self.results['dice_et'].append(dice_batch[2].item())
                self.results['mean_iou'].append(mean_iou)
                self.results['hausdorff'].append(hausdorff)
                self.results['precision'].append(additional_metrics['precision'])
                self.results['recall'].append(additional_metrics['recall'])
                self.results['f1'].append(additional_metrics['f1'])
                
                # Save predictions if requested
                if self.save_predictions:
                    for i, (pred, label) in enumerate(zip(val_outputs_processed, val_labels_processed)):
                        pred_path = self.output_dir / f"pred_batch{batch_idx}_sample{i}.npy"
                        label_path = self.output_dir / f"label_batch{batch_idx}_sample{i}.npy"
                        np.save(pred_path, pred.cpu().numpy())
                        np.save(label_path, label.cpu().numpy())
                
                # Reset metrics for next batch
                self.dice_metric.reset()
                self.dice_metric_batch.reset()
                self.jaccard_metric.reset()
                self.hausdorff_metric.reset()
                
                # Track timing
                batch_time = time.time() - start_time
                total_time += batch_time
                num_samples += val_inputs.size(0)
                
                # Log progress for every batch
                total_batches = self.max_batches if self.max_batches else len(self.val_loader)
                print(f"Batch {batch_idx}/{total_batches}: "
                      f"Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, "
                      f"Time={batch_time:.2f}s")
        
        # Compute final statistics
        final_results = {}
        for metric_name, values in self.results.items():
            if values:
                # Filter out NaN values for statistics
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    final_results[f"{metric_name}_mean"] = np.mean(valid_values)
                    final_results[f"{metric_name}_std"] = np.std(valid_values)
                    final_results[f"{metric_name}_median"] = np.median(valid_values)
                else:
                    final_results[f"{metric_name}_mean"] = float('nan')
                    final_results[f"{metric_name}_std"] = float('nan')
                    final_results[f"{metric_name}_median"] = float('nan')
        
        # Add timing information
        final_results['avg_time_per_sample'] = total_time / num_samples if num_samples > 0 else 0
        final_results['total_validation_time'] = total_time
        final_results['num_samples'] = num_samples
        
        return final_results
        
    def print_results(self, results: Dict[str, float]):
        """Print validation results in a formatted manner."""
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)
        
        print(f"Dataset: {self.dataset}")
        print(f"Checkpoint: {os.path.basename(self.checkpoint_path)}")
        print(f"Number of samples: {results['num_samples']}")
        print(f"TTA enabled: {self.use_tta}")
        if self.use_tta:
            print(f"TTA merge mode: {self.tta_merge_mode}")
        print(f"ROI size: {self.roi_size}")
        print(f"Overlap: {self.overlap}")
        print(f"Threshold: {self.threshold}")
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        
        # Main metrics
        dice_mean = results.get('mean_dice_mean', float('nan'))
        dice_std = results.get('mean_dice_std', float('nan'))
        print(f"Mean Dice Score: {dice_mean:.4f} ± {dice_std:.4f}")
        
        # Individual class Dice scores
        tc_mean = results.get('dice_tc_mean', float('nan'))
        tc_std = results.get('dice_tc_std', float('nan'))
        print(f"TC Dice Score:   {tc_mean:.4f} ± {tc_std:.4f}")
        
        wt_mean = results.get('dice_wt_mean', float('nan'))
        wt_std = results.get('dice_wt_std', float('nan'))
        print(f"WT Dice Score:   {wt_mean:.4f} ± {wt_std:.4f}")
        
        et_mean = results.get('dice_et_mean', float('nan'))
        et_std = results.get('dice_et_std', float('nan'))
        print(f"ET Dice Score:   {et_mean:.4f} ± {et_std:.4f}")
        
        # Other metrics
        iou_mean = results.get('mean_iou_mean', float('nan'))
        iou_std = results.get('mean_iou_std', float('nan'))
        print(f"Mean IoU:        {iou_mean:.4f} ± {iou_std:.4f}")
        
        hausdorff_mean = results.get('hausdorff_mean', float('nan'))
        hausdorff_std = results.get('hausdorff_std', float('nan'))
        print(f"Hausdorff Dist.: {hausdorff_mean:.4f} ± {hausdorff_std:.4f}")
        
        precision_mean = results.get('precision_mean', float('nan'))
        precision_std = results.get('precision_std', float('nan'))
        print(f"Precision:       {precision_mean:.4f} ± {precision_std:.4f}")
        
        recall_mean = results.get('recall_mean', float('nan'))
        recall_std = results.get('recall_std', float('nan'))
        print(f"Recall:          {recall_mean:.4f} ± {recall_std:.4f}")
        
        f1_mean = results.get('f1_mean', float('nan'))
        f1_std = results.get('f1_std', float('nan'))
        print(f"F1 Score:        {f1_mean:.4f} ± {f1_std:.4f}")
        
        # Timing
        avg_time = results.get('avg_time_per_sample', 0)
        total_time = results.get('total_validation_time', 0)
        print(f"\nTIMING:")
        print("-" * 40)
        print(f"Avg time per sample: {avg_time:.2f}s")
        print(f"Total validation time: {total_time:.2f}s")
        
        print("="*80)
        
        # Log to WandB
        if self.log_to_wandb:
            wandb.log(results)
            
    def save_results(self, results: Dict[str, float], filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"validation_results_{self.dataset}_{timestamp}.json"
            
        results_path = self.output_dir / filename
        
        # Include configuration in results
        config = {
            'checkpoint_path': self.checkpoint_path,
            'dataset': self.dataset,
            'use_tta': self.use_tta,
            'tta_merge_mode': self.tta_merge_mode,
            'roi_size': self.roi_size,
            'overlap': self.overlap,
            'threshold': self.threshold,
            'batch_size': self.batch_size
        }
        
        full_results = {
            'config': config,
            'results': results,
            'raw_data': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
            
        print(f"\nResults saved to: {results_path}")
        
    def run(self):
        """Run the complete validation pipeline."""
        self.setup()
        results = self.validate()
        self.print_results(results)
        
        if self.save_predictions or True:  # Always save results
            self.save_results(results)
            
        if self.log_to_wandb:
            wandb.finish()
            
        return results

def main():
    """Main function for standalone validation."""
    parser = argparse.ArgumentParser(description="Standalone SwinUNETR Validation with TTA")
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint (.ckpt or .pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing dataset.json')
    
    # Dataset and basic settings
    parser.add_argument('--dataset', type=str, default='brats2023', 
                        choices=['brats2021', 'brats2023'],
                        help='Dataset format (default: brats2023)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for validation (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=48,
                        help='Model feature size (default: 48)')
    parser.add_argument('--use_v2', action='store_true', default=True,
                        help='Use SwinUNETR V2 (default: True)')
    parser.add_argument('--downsample', type=str, default='mergingv2',
                        help='Downsample method (default: mergingv2)')
    parser.add_argument('--use_modality_attention', action='store_true',
                        help='Use modality attention module (default: False)')
    
    # Inference parameters
    parser.add_argument('--roi_size', type=int, nargs=3, default=[96, 96, 96],
                        help='ROI size for sliding window inference (default: 96 96 96)')
    parser.add_argument('--sw_batch_size', type=int, default=1,
                        help='Sliding window batch size (default: 1)')
    parser.add_argument('--overlap', type=float, default=0.7,
                        help='Sliding window overlap (default: 0.7)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for post-processing (default: 0.5)')
    
    # TTA parameters
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Enable Test Time Augmentation (default: True)')
    parser.add_argument('--tta_merge_mode', type=str, default='mean',
                        choices=['mean', 'median', 'max'],
                        help='TTA ensemble method (default: mean)')
    
    # Output and logging
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to validate (default: None - all batches)')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Output directory for results (default: ./validation_results)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction arrays (default: False)')
    parser.add_argument('--log_to_wandb', action='store_true',
                        help='Log results to Weights & Biases (default: False)')
    parser.add_argument('--wandb_project', type=str, default='swinunetr-validation',
                        help='WandB project name (default: swinunetr-validation)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Create validation pipeline
    pipeline = StandaloneValidationPipeline(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        threshold=args.threshold,
        use_tta=args.use_tta,
        tta_merge_mode=args.tta_merge_mode,
        device=args.device,
        feature_size=args.feature_size,
        use_v2=args.use_v2,
        downsample=args.downsample,
        use_modality_attention=args.use_modality_attention,
        max_batches=args.max_batches,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project
    )
    
    # Run validation
    results = pipeline.run()
    
    return results

if __name__ == "__main__":
    main()