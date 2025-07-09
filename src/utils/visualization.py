import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
import os

def visualize_batch(batch, model, device, slice_idx=None):
    """Visualize a batch of images and their predictions."""
    model.eval()
    with torch.no_grad():
        # Get input images and labels
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Get model predictions
        outputs = model(images)
        outputs = [AsDiscrete()(i) for i in decollate_batch(outputs)]
        labels = [AsDiscrete()(i) for i in decollate_batch(labels)]
        
        # Convert to numpy for visualization
        images = images.cpu().numpy()
        outputs = [o.cpu().numpy() for o in outputs]
        labels = [l.cpu().numpy() for l in labels]
        
        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = images.shape[2] // 2
        
        # Create figure
        fig, axes = plt.subplots(len(images), 4, figsize=(20, 5*len(images)))
        if len(images) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (img, pred, label) in enumerate(zip(images, outputs, labels)):
            # Show T1c image
            axes[idx, 0].imshow(img[0, :, :, slice_idx], cmap='gray')
            axes[idx, 0].set_title('T1c Input')
            axes[idx, 0].axis('off')
            
            # Show prediction
            axes[idx, 1].imshow(pred[0, :, :, slice_idx], cmap='viridis')
            axes[idx, 1].set_title('Prediction')
            axes[idx, 1].axis('off')
            
            # Show ground truth
            axes[idx, 2].imshow(label[0, :, :, slice_idx], cmap='viridis')
            axes[idx, 2].set_title('Ground Truth')
            axes[idx, 2].axis('off')
            
            # Show overlay
            axes[idx, 3].imshow(img[0, :, :, slice_idx], cmap='gray')
            axes[idx, 3].imshow(pred[0, :, :, slice_idx], cmap='viridis', alpha=0.5)
            axes[idx, 3].set_title('Overlay')
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        return fig

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(train_metrics, label='Training Dice')
    ax2.plot(val_metrics, label='Validation Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def visualize_predictions(model, val_loader, device, num_samples=4, save_dir=None):
    """Visualize predictions on validation set."""
    model.eval()
    batch = next(iter(val_loader))
    fig = visualize_batch(batch, model, device)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'predictions.png'))
    
    return fig

# Plotting train vs validation metrics
plt.figure("train", (12, 6))

# Plot 1: Epoch Average Loss
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(model.avg_train_loss_values))]
y = model.avg_train_loss_values
plt.xlabel("Epoch")
plt.plot(x, y, color="red", label="Train Loss")
plt.legend()

# Plot 2: Train Mean Dice
plt.subplot(1, 2, 2)
plt.title("Train Mean Dice")
x = [i + 1 for i in range(len(model.train_metric_values))]
y = model.train_metric_values
plt.xlabel("Epoch")
plt.plot(x, y, color="green", label="Train Dice")
plt.legend()

plt.show()

# Plotting dice metrics for different categories (TC, WT, ET)
plt.figure("train", (18, 6))

# Plot 1: Train Mean Dice TC
plt.subplot(1, 3, 1)
plt.title("Train Mean Dice TC")
x = [i + 1 for i in range(len(model.train_metric_values_tc))]
y = model.train_metric_values_tc
plt.xlabel("Epoch")
plt.plot(x, y, color="blue", label="Train TC Dice")
plt.legend()

# Plot 2: Train Mean Dice WT
plt.subplot(1, 3, 2)
plt.title("Train Mean Dice WT")
x = [i + 1 for i in range(len(model.train_metric_values_wt))]
y = model.train_metric_values_wt
plt.xlabel("Epoch")
plt.plot(x, y, color="brown", label="Train WT Dice")
plt.legend()

# Plot 3: Train Mean Dice ET
plt.subplot(1, 3, 3)
plt.title("Train Mean Dice ET")
x = [i + 1 for i in range(len(model.train_metric_values_et))]
y = model.train_metric_values_et
plt.xlabel("Epoch")
plt.plot(x, y, color="purple", label="Train ET Dice")
plt.legend()

plt.show()



plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(model.epoch_loss_values))]
y = model.epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values))]
y = model.metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.show()

plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values_tc))]
y = model.metric_values_tc
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values_wt))]
y = model.metric_values_wt
plt.xlabel("epoch")
plt.plot(x, y, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
x = [model.hparams.val_interval * (i + 1) for i in range(len(model.metric_values_et))]
y = model.metric_values_et
plt.xlabel("epoch")
plt.plot(x, y, color="purple")
plt.show()