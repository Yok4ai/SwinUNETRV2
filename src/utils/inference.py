"""
Standalone inference pipeline for SwinUNETR V2 model.
This script loads a trained checkpoint and runs inference on a test dataset,
generating all metrics and predictions similar to the validation pipeline.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import json
import torch
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.utils import first
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# Import project modules
from src.data.augmentations import get_transforms
from src.data.convert_labels import ConvertLabels
from monai.networks.nets import SwinUNETR
from kaggle_setup import prepare_brats_data

# Suppress warnings
warnings.filterwarnings('ignore')

class InferenceEngine:
    """
    Standalone inference engine for SwinUNETR V2 model.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.jaccard_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
        
        # Post-processing transforms
        self.post_trans = Compose([
            Activations(sigmoid=True), 
            AsDiscrete(threshold=args.threshold)
        ])
        
        # Storage for results
        self.results = {
            'dice_scores': [],
            'dice_tc': [],
            'dice_wt': [],
            'dice_et': [],
            'iou_scores': [],
            'hausdorff_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'f1_scores': [],
            'case_names': []
        }
        
    def setup_dataset(self) -> DataLoader:
        """Setup the test dataset and dataloader."""
        print("Setting up dataset...")
        
        # Prepare data if needed
        if self.args.prepare_data:
            print(f"Preparing BraTS {self.args.dataset} data...")
            prepare_brats_data(
                input_dir=self.args.input_dir,
                output_dir=self.args.output_dir,
                dataset=self.args.dataset
            )
            data_dir = self.args.output_dir
        else:
            data_dir = self.args.input_dir
            
        # Load dataset.json
        dataset_path = os.path.join(data_dir, "dataset.json")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        with open(dataset_path) as f:
            datalist = json.load(f)["training"]
        
        # Get transforms
        _, val_transforms = get_transforms(
            img_size=self.args.img_size, 
            dataset=self.args.dataset
        )
        
        # Use validation split or full dataset based on user choice
        if self.args.use_val_split:
            # Use same split as training (80/20)
            train_files, val_files = train_test_split(datalist, test_size=0.2, random_state=42)
            test_files = val_files
            print(f"Using validation split: {len(test_files)} cases")
        else:
            # Use full dataset
            test_files = datalist
            print(f"Using full dataset: {len(test_files)} cases")
        
        # Create dataset and dataloader
        test_ds = Dataset(data=test_files, transform=val_transforms)
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers
        )
        
        return test_loader
    
    def load_model(self) -> SwinUNETR:
        """Load the trained model from checkpoint."""
        print(f"Loading model from: {self.args.checkpoint_path}")
        
        if not os.path.exists(self.args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.args.checkpoint_path}")
        
        # Initialize standard MONAI SwinUNETR model
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=self.args.feature_size,
            depths=self.args.depths,
            num_heads=self.args.num_heads,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            spatial_dims=3,
            downsample=self.args.downsample,
            use_v2=self.args.use_v2
        )
        
        # Load state dict
        if self.args.checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint - extract only the SwinUNETR model weights
            checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)
            # Filter to only include SwinUNETR model weights (remove Lightning wrapper)
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    # Remove the 'model.' prefix to match direct SwinUNETR
                    new_key = key[6:]
                    model_state_dict[new_key] = value
            model.load_state_dict(model_state_dict, strict=False)
        else:
            # PyTorch state dict
            state_dict = torch.load(self.args.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def compute_metrics(self, outputs: List[torch.Tensor], labels: List[torch.Tensor]) -> Dict[str, float]:
        """Compute comprehensive metrics for a batch."""
        # Stack outputs and labels
        outputs_stacked = torch.stack(outputs) if isinstance(outputs, list) else outputs
        labels_stacked = torch.stack(labels) if isinstance(labels, list) else labels
        
        outputs_stacked = outputs_stacked.float()
        labels_stacked = labels_stacked.float()
        
        # Compute precision, recall, F1
        outputs_flat = outputs_stacked.view(outputs_stacked.size(0), -1)
        labels_flat = labels_stacked.view(labels_stacked.size(0), -1)
        
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
    
    def run_inference(self, model: SwinUNETR, test_loader: DataLoader):
        """Run inference on the test dataset."""
        print("Starting inference...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs, labels = batch["image"], batch["label"]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
                
                # Sliding window inference
                outputs = sliding_window_inference(
                    inputs, 
                    roi_size=self.args.roi_size, 
                    sw_batch_size=self.args.sw_batch_size,
                    predictor=model, 
                    overlap=self.args.overlap
                )
                
                # Apply post-processing
                outputs_post = [self.post_trans(i) for i in decollate_batch(outputs)]
                labels_post = decollate_batch(labels)
                
                # Compute metrics for this batch
                self.dice_metric(y_pred=outputs_post, y=labels_post)
                self.dice_metric_batch(y_pred=outputs_post, y=labels_post)
                self.jaccard_metric(y_pred=outputs_post, y=labels_post)
                self.hausdorff_metric(y_pred=outputs_post, y=labels_post)
                
                # Get batch metrics
                dice_score = self.dice_metric.aggregate().item()
                dice_batch = self.dice_metric_batch.aggregate()
                iou_score = self.jaccard_metric.aggregate().item()
                
                # Handle Hausdorff distance
                hausdorff_values = self.hausdorff_metric.aggregate(reduction='none')
                if not isinstance(hausdorff_values, torch.Tensor):
                    hausdorff_values = torch.tensor(hausdorff_values)
                valid = torch.isfinite(hausdorff_values)
                hausdorff_score = hausdorff_values[valid].mean().item() if valid.any() else float('nan')
                
                # Compute additional metrics
                additional_metrics = self.compute_metrics(outputs_post, labels_post)
                
                # Store results
                self.results['dice_scores'].append(dice_score)
                self.results['dice_tc'].append(dice_batch[0].item())
                self.results['dice_wt'].append(dice_batch[1].item())
                self.results['dice_et'].append(dice_batch[2].item())
                self.results['iou_scores'].append(iou_score)
                self.results['hausdorff_scores'].append(hausdorff_score)
                self.results['precision_scores'].append(additional_metrics['precision'])
                self.results['recall_scores'].append(additional_metrics['recall'])
                self.results['f1_scores'].append(additional_metrics['f1'])
                self.results['case_names'].append(f"case_{batch_idx}")
                
                # Store predictions if requested
                if self.args.save_predictions:
                    all_predictions.extend(outputs_post)
                    all_labels.extend(labels_post)
                
                # Print batch results
                print(f"  Dice: {dice_score:.4f}, TC: {dice_batch[0].item():.4f}, "
                      f"WT: {dice_batch[1].item():.4f}, ET: {dice_batch[2].item():.4f}")
                
                # Reset metrics for next batch
                self.dice_metric.reset()
                self.dice_metric_batch.reset()
                self.jaccard_metric.reset()
                self.hausdorff_metric.reset()
        
        # Save predictions if requested
        if self.args.save_predictions:
            self.save_predictions(all_predictions, all_labels)
        
        print("Inference completed!")
    
    def save_predictions(self, predictions: List[torch.Tensor], labels: List[torch.Tensor]):
        """Save predictions to disk."""
        print("Saving predictions...")
        
        pred_dir = os.path.join(self.args.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            pred_path = os.path.join(pred_dir, f"prediction_{i}.npy")
            label_path = os.path.join(pred_dir, f"label_{i}.npy")
            
            np.save(pred_path, pred.cpu().numpy())
            np.save(label_path, label.cpu().numpy())
        
        print(f"Predictions saved to: {pred_dir}")
    
    def generate_report(self):
        """Generate comprehensive inference report."""
        print("\n" + "="*60)
        print("INFERENCE REPORT")
        print("="*60)
        
        # Calculate statistics
        metrics = {
            'Mean Dice': np.mean(self.results['dice_scores']),
            'Mean Dice TC': np.mean(self.results['dice_tc']),
            'Mean Dice WT': np.mean(self.results['dice_wt']),
            'Mean Dice ET': np.mean(self.results['dice_et']),
            'Mean IoU': np.mean(self.results['iou_scores']),
            'Mean Hausdorff': np.nanmean(self.results['hausdorff_scores']),
            'Mean Precision': np.mean(self.results['precision_scores']),
            'Mean Recall': np.mean(self.results['recall_scores']),
            'Mean F1': np.mean(self.results['f1_scores']),
        }
        
        std_metrics = {
            'Std Dice': np.std(self.results['dice_scores']),
            'Std Dice TC': np.std(self.results['dice_tc']),
            'Std Dice WT': np.std(self.results['dice_wt']),
            'Std Dice ET': np.std(self.results['dice_et']),
            'Std IoU': np.std(self.results['iou_scores']),
            'Std Hausdorff': np.nanstd(self.results['hausdorff_scores']),
            'Std Precision': np.std(self.results['precision_scores']),
            'Std Recall': np.std(self.results['recall_scores']),
            'Std F1': np.std(self.results['f1_scores']),
        }
        
        # Print metrics
        print(f"Dataset: {self.args.dataset}")
        print(f"Checkpoint: {self.args.checkpoint_path}")
        print(f"Number of cases: {len(self.results['dice_scores'])}")
        print(f"ROI size: {self.args.roi_size}")
        print(f"Overlap: {self.args.overlap}")
        print(f"Threshold: {self.args.threshold}")
        print()
        
        print("PERFORMANCE METRICS:")
        print("-" * 40)
        for metric, value in metrics.items():
            std_key = metric.replace('Mean', 'Std')
            std_value = std_metrics.get(std_key, 0.0)
            print(f"{metric:20s}: {value:.4f} ± {std_value:.4f}")
        
        print("\nDETAILED DICE SCORES:")
        print("-" * 40)
        print(f"{'Case':<10} {'Mean':<8} {'TC':<8} {'WT':<8} {'ET':<8}")
        print("-" * 40)
        for i, case in enumerate(self.results['case_names']):
            print(f"{case:<10} {self.results['dice_scores'][i]:<8.4f} "
                  f"{self.results['dice_tc'][i]:<8.4f} {self.results['dice_wt'][i]:<8.4f} "
                  f"{self.results['dice_et'][i]:<8.4f}")
        
        # Save report to file
        report_path = os.path.join(self.args.output_dir, "inference_report.txt")
        with open(report_path, 'w') as f:
            f.write("INFERENCE REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Dataset: {self.args.dataset}\n")
            f.write(f"Checkpoint: {self.args.checkpoint_path}\n")
            f.write(f"Number of cases: {len(self.results['dice_scores'])}\n")
            f.write(f"ROI size: {self.args.roi_size}\n")
            f.write(f"Overlap: {self.args.overlap}\n")
            f.write(f"Threshold: {self.args.threshold}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            for metric, value in metrics.items():
                std_key = metric.replace('Mean', 'Std')
                std_value = std_metrics.get(std_key, 0.0)
                f.write(f"{metric:20s}: {value:.4f} ± {std_value:.4f}\n")
            
            f.write(f"\nDETAILED DICE SCORES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Case':<10} {'Mean':<8} {'TC':<8} {'WT':<8} {'ET':<8}\n")
            f.write("-" * 40 + "\n")
            for i, case in enumerate(self.results['case_names']):
                f.write(f"{case:<10} {self.results['dice_scores'][i]:<8.4f} "
                       f"{self.results['dice_tc'][i]:<8.4f} {self.results['dice_wt'][i]:<8.4f} "
                       f"{self.results['dice_et'][i]:<8.4f}\n")
        
        print(f"\nReport saved to: {report_path}")
        
        # Generate plots if requested
        if self.args.save_plots:
            self.generate_plots()
        
        print("="*60)
    
    def generate_plots(self):
        """Generate visualization plots."""
        print("Generating plots...")
        
        plots_dir = os.path.join(self.args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Dice scores distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(self.results['dice_scores'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Mean Dice Scores')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        labels = ['TC', 'WT', 'ET']
        means = [np.mean(self.results['dice_tc']), np.mean(self.results['dice_wt']), np.mean(self.results['dice_et'])]
        stds = [np.std(self.results['dice_tc']), np.std(self.results['dice_wt']), np.std(self.results['dice_et'])]
        plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        plt.ylabel('Dice Score')
        plt.title('Mean Dice Scores by Class')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.results['dice_scores'], 'o-', alpha=0.7)
        plt.xlabel('Case Index')
        plt.ylabel('Dice Score')
        plt.title('Dice Scores Across Cases')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        metrics_plot = ['Precision', 'Recall', 'F1']
        means_plot = [np.mean(self.results['precision_scores']), 
                     np.mean(self.results['recall_scores']), 
                     np.mean(self.results['f1_scores'])]
        stds_plot = [np.std(self.results['precision_scores']), 
                    np.std(self.results['recall_scores']), 
                    np.std(self.results['f1_scores'])]
        plt.bar(metrics_plot, means_plot, yerr=stds_plot, capsize=5, alpha=0.7, edgecolor='black')
        plt.ylabel('Score')
        plt.title('Additional Metrics')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'inference_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {plots_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SwinUNETR V2 Standalone Inference Pipeline")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing BraTS data or dataset.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="Output directory for results")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, choices=["brats2021", "brats2023"], default="brats2023",
                       help="Dataset version (brats2021 or brats2023)")
    parser.add_argument("--prepare_data", action="store_true",
                       help="Prepare BraTS data and create dataset.json")
    parser.add_argument("--use_val_split", action="store_true",
                       help="Use validation split (80/20) instead of full dataset")
    
    # Model arguments
    parser.add_argument("--img_size", type=int, default=96,
                       help="Image size for processing")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96],
                       help="ROI size for sliding window inference")
    parser.add_argument("--sw_batch_size", type=int, default=2,
                       help="Batch size for sliding window inference")
    parser.add_argument("--overlap", type=float, default=0.7,
                       help="Overlap for sliding window inference")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for converting probabilities to binary")
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Model architecture arguments
    parser.add_argument("--feature_size", type=int, default=48,
                       help="Feature size for SwinUNETR")
    parser.add_argument("--use_v2", action="store_true", default=True,
                       help="Use SwinUNETR V2")
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 2, 2],
                       help="Depths for SwinUNETR layers")
    parser.add_argument("--num_heads", type=int, nargs=4, default=[3, 6, 12, 24],
                       help="Number of heads for SwinUNETR layers")
    parser.add_argument("--downsample", type=str, default="mergingv2",
                       help="Downsampling method")
    parser.add_argument("--use_modality_attention", action="store_true",
                       help="Use modality attention module")
    
    # Training-related arguments (for model initialization)
    parser.add_argument("--max_epochs", type=int, default=50,
                       help="Maximum epochs (for model init)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (for model init)")
    parser.add_argument("--use_class_weights", action="store_true",
                       help="Use class weights")
    parser.add_argument("--class_weights", type=float, nargs=3, default=[3.0, 1.0, 5.0],
                       help="Class weights for TC, WT, ET")
    parser.add_argument("--loss_type", type=str, default="dice",
                       help="Loss type (for model init)")
    
    # Output arguments
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save predictions to disk")
    parser.add_argument("--save_plots", action="store_true",
                       help="Generate and save visualization plots")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    
    return parser.parse_args()

def main():
    """Main inference function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("SwinUNETR V2 Standalone Inference Pipeline")
    print("="*50)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)
    
    # Initialize inference engine
    engine = InferenceEngine(args)
    
    # Setup dataset
    test_loader = engine.setup_dataset()
    
    # Load model
    model = engine.load_model()
    
    # Run inference
    engine.run_inference(model, test_loader)
    
    # Generate report
    engine.generate_report()
    
    print("\nInference pipeline completed successfully!")

if __name__ == "__main__":
    main()