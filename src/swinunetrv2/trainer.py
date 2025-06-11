# Training setup
max_epochs = 30
train_ds = train_ds
val_ds = val_ds
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=False)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, persistent_workers=False)

# set up early stopping - monitoring val_mean_dice for better performance tracking
early_stop_callback = EarlyStopping(
   monitor="val_mean_dice",
   min_delta=0.00,
   patience=7,  # Increased patience for V2 training
   verbose=True,
   mode='max'  # Changed to max since we want higher dice scores
)
# stop training after 11 hours
timer_callback = Timer(duration="00:11:00:00")

# Initialize wandb logger
wandb.init(project="brain-tumor-segmentation", name="swinunetr-v2-brats23")  # Updated to reflect SwinUNETR-V2

# Setup your logger in the Trainer
wandb_logger = WandbLogger()

# Initialize and train the model
model = BrainTumorSegmentation(train_loader, val_loader, max_epochs=max_epochs)
trainer = pl.Trainer(max_epochs=max_epochs,
                     devices=1,
                     accelerator="gpu",
                     precision = '16-mixed',
                     gradient_clip_val=1.0,  # Gradient clipping
                     log_every_n_steps=1,
                     # val_check_interval=1.0,
                     callbacks=[early_stop_callback, timer_callback],
                     limit_val_batches = 5,
                     check_val_every_n_epoch=1,
                     logger=wandb_logger, 
                    )

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(f"Train completed, best_metric: {model.best_metric:.4f} at epoch: {model.best_metric_epoch}.")

import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from utils.visualization import visualize_predictions, plot_training_curves
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4, epochs=100, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        
        # Loss and optimizer
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Create output directory
        self.output_dir = '/kaggle/working/outputs'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        self.dice_metric.reset()
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Get data
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.dice_metric(y_pred=outputs, y=labels)
        
        # Calculate average loss and metric
        avg_loss = epoch_loss / len(self.train_loader)
        avg_metric = self.dice_metric.aggregate().item()
        
        return avg_loss, avg_metric
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        self.dice_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Get data
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                outputs = sliding_window_inference(
                    images, 
                    roi_size=(96, 96, 96), 
                    sw_batch_size=4, 
                    predictor=self.model
                )
                loss = self.loss_function(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                self.dice_metric(y_pred=outputs, y=labels)
        
        # Calculate average loss and metric
        avg_loss = val_loss / len(self.val_loader)
        avg_metric = self.dice_metric.aggregate().item()
        
        return avg_loss, avg_metric
    
    def train(self):
        best_val_metric = -1
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training
            train_loss, train_metric = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metric)
            
            # Validation
            val_loss, val_metric = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metric)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metric:.4f}")
            
            # Save best model
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            
            # Visualize predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                fig = visualize_predictions(
                    self.model, 
                    self.val_loader, 
                    self.device,
                    save_dir=self.output_dir
                )
                plt.close(fig)
            
            # Plot training curves
            fig = plot_training_curves(
                self.train_losses,
                self.val_losses,
                self.train_metrics,
                self.val_metrics,
                save_path=os.path.join(self.output_dir, 'training_curves.png')
            )
            plt.close(fig)
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'final_model.pth'))
        
        # Final visualization
        fig = visualize_predictions(
            self.model, 
            self.val_loader, 
            self.device,
            save_dir=self.output_dir
        )
        plt.close(fig)