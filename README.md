# SwinUNETR V2

A PyTorch implementation of SwinUNETR V2 for medical image segmentation.

## Installation

### On Kaggle

Clone the repository and install the package locally:

```python
!if [ -d "SwinUNETRV2" ]; then cd SwinUNETRV2 && git pull && cd ..; else git clone https://github.com/Yok4ai/SwinUNETRV2.git; fi
!pip install -q ./SwinUNETRV2
```

This will automatically install all required dependencies including:
- MONAI with nibabel, tqdm, and einops
- Matplotlib
- Einops
- And other required packages

### Local Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Yok4ai/SwinUNETRV2.git
cd SwinUNETRV2
pip install -e .
```

## Project Structure

```
SWINUNETRV2
├── src/
│     ├── models/         # Model architecture
│     ├── data/          # Data loading and augmentation
│     ├── utils/         # Utility functions
├── setup.py              # Package setup file
├── kaggle_run.py         # Main entry point for training (use this script)
├── main.py
└── requirements.txt      # Project dependencies
```

## Execution Flow

The training pipeline is orchestrated as follows:

1. **kaggle_run.py** (or run locally):
    - Sets up the environment (e.g., prepares data, applies Kaggle-specific settings).
    - Defines all parameters for data, model, training, validation, and inference in a single `args` block.
    - Calls the `main` function from `main.py` with these parameters.

2. **main.py**:
    - Receives the `args` namespace and coordinates the full training pipeline:
      1. **Data Augmentation**: Calls `get_transforms` from `src/data/augmentations.py` to set up MONAI transforms for training and validation.
      2. **Data Loading**: Calls `get_dataloaders` from `src/data/dataloader.py` to create PyTorch DataLoaders using the transforms and dataset split.
      3. **Model & Trainer Setup**: Calls `setup_training` from `src/models/trainer.py` to initialize the SwinUNETR model (via `BrainTumorSegmentation` in `src/models/pipeline.py`) and the PyTorch Lightning Trainer with all callbacks and logging.
      4. **Training**: Calls `train_model` from `src/models/trainer.py` to start the training loop, validation, and checkpointing.

3. **Model Details**:
    - The SwinUNETR model is defined in `src/models/swinunetr.py` and wrapped by `BrainTumorSegmentation` in `src/models/pipeline.py` for training, validation, and logging.
    - Training and validation metrics, early stopping, and checkpointing are handled by PyTorch Lightning.

**Summary:**
- Edit parameters in `kaggle_run.py`.
- Run `kaggle_run.py` (on Kaggle or locally).
- The script will handle all steps: data transforms, loading, model setup, and training.

## Usage

### Running on Kaggle

1. Install the package (see above).
2. Run the training script:

```python
!python ./SwinUNETRV2/kaggle_run.py
```

This will use the optimized configuration in `kaggle_run.py` for training.

#### Customizing Parameters

You can customize the training, model, and data parameters by editing the `args` block in `kaggle_run.py`. The parameters are grouped as follows:

- **Data parameters**: `input_dir`, `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`
- **Model parameters**: `img_size`, `in_channels`, `out_channels`, `feature_size`, `depths`, `num_heads`, `downsample`, `use_v2`
- **Training parameters**: `learning_rate`, `weight_decay`, `epochs`, `accelerator`, `devices`, `precision`, `strategy`, `log_every_n_steps`, `enable_checkpointing`, `benchmark`, `profiler`, `use_amp`, `gradient_clip_val`
- **Validation settings**: `val_interval`, `save_interval`, `early_stopping_patience`, `limit_val_batches`
- **Inference parameters**: `roi_size`, `sw_batch_size`, `overlap`

To change a parameter, simply edit its value in the `argparse.Namespace` in `kaggle_run.py`. For example, to use a batch size of 8 and train for 20 epochs:

```python
args = argparse.Namespace(
    # ...
    batch_size=8,
    epochs=20,
    # ...
)
```

You can adjust any of the parameters in this block to fit your dataset, hardware, or experiment needs.

### Running Locally

You can run the entire pipeline using the main script:

```bash
python kaggle_run.py
```

Or, for advanced usage, you can use the main module directly (not recommended for Kaggle):

```bash
python -m swinunetrv2.main \
    --input_dir /path/to/input/data \
    --output_dir /path/to/output/data \
    --batch_size 8 \
    --img_size 96 \
    --in_channels 4 \
    --out_channels 3 \
    --feature_size 12 \
    --learning_rate 1e-4 \
    --epochs 100
```

### Using as a Package

You can also import and use individual components:

```python
from swinunetrv2.models.swinunetr import SwinUNETR
from swinunetrv2.models.trainer import Trainer

# Initialize model
model = SwinUNETR(...)

# Train model
trainer = Trainer(model, ...)
trainer.train()
```

## License

MIT License 