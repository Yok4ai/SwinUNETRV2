# Import necessary modules
from swinunetrv2.kaggle_setup import setup_kaggle_notebook
from swinunetrv2.main import main
import argparse

# Setup the environment and prepare data
output_dir = setup_kaggle_notebook()
print(f"Dataset prepared in: {output_dir}")

# Configure training parameters
args = argparse.Namespace(
    # Data parameters
    input_dir='/kaggle/working',  # Just the directory path
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    
    # Model parameters
    img_size=128,
    in_channels=4,  # 4 modalities for BraTS (t1c, t1n, t2f, t2w)
    out_channels=3,  # 3 tumor regions for BraTS
    feature_size=48,
    embed_dim=48,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    decoder_embed_dim=256,
    patch_size=4,
    drop_rate=0.1,
    attn_drop_rate=0.1,
    use_checkpoint=True,  # Enable checkpointing for memory efficiency
    
    # Training parameters
    learning_rate=2e-3,
    weight_decay=1e-4,
    epochs=50,
    warmup_epochs=5,
    device='cuda',
    use_amp=True,
    gradient_clip_val=1.0,
    
    # Validation settings
    val_interval=1,
    save_interval=10,
    early_stopping_patience=15,
    
    # Inference parameters
    roi_size=[128, 128, 128],
    sw_batch_size=2,
    overlap=0.25
)

# Start training
main(args) 