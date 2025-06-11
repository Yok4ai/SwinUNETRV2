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