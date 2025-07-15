# SwinUNETR++

An advanced PyTorch implementation of SwinUNETR++ for brain tumor segmentation with comprehensive loss function optimization, adaptive scheduling, and local minima escape strategies.

## Installation

### On Kaggle 

Clone the repository (Experimental Branch) and install the package locally:

```python
!if [ -d "SwinUNETRV2" ]; then \
    cd SwinUNETRV2 && \
    git fetch origin && \
    git checkout experimental && \
    git pull origin experimental && \
    cd ..; \
else \
    git clone --branch experimental https://github.com/Yok4ai/SwinUNETRV2.git; \
fi

# Install the package
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

## 🎯 Key Features

### Advanced Loss Functions (14 Total)
- **Basic**: Dice, DiceCE, DiceFocal, Generalized Dice, Focal, Tversky, Hausdorff
- **Hybrid**: GDL+Focal+Tversky, Dice+Hausdorff combinations
- **Adaptive**: Structure-boundary scheduling, progressive hybrid, complexity cascade, dynamic performance-based
- **Local Minima Escape**: Warm restarts and plateau detection

### SOTA Optimization
- **Adaptive Scheduling**: Automatic curriculum learning with configurable phase transitions
- **Dynamic Loss Weighting**: Performance-based adaptation during training
- **Multiple Schedule Types**: Linear, exponential, cosine weight transitions
- **BraTS Specialized**: Optimized for brain tumor segmentation challenges

### Easy CLI Interface
- **30+ Parameters**: Comprehensive configuration options
- **Quick Start to Competition**: From simple Dice to SOTA adaptive strategies
- **Example Commands**: Documented usage patterns for all scenarios

## Project Structure

```
SwinUNETR++
├── src/
│     ├── models/         # Enhanced model architecture with adaptive losses
│     ├── data/          # Data loading and augmentation (BraTS 2021/2023)
│     ├── utils/         # Visualization and utility functions
├── docs/                 # Comprehensive documentation
│     ├── loss.md        # ⭐ Complete loss function guide
│     ├── architecture.md # Pipeline architecture overview
│     └── ...            # Additional documentation
├── kaggle_run.py         # Main CLI entry point (30+ parameters)
├── main.py              # Core training orchestration
├── setup.py             # Package setup
└── requirements.txt     # Dependencies
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
    - The SwinUNETR++ model is defined in `src/models/swinunetr.py` with enhanced features in `swinunetrplus.py` 
    - Wrapped by `BrainTumorSegmentation` in `src/models/pipeline.py` with 14 loss functions and adaptive scheduling
    - Advanced optimization with warm restarts, plateau detection, and automatic curriculum learning

**Summary:**
- Edit parameters in `kaggle_run.py`.
- Run `kaggle_run.py` (on Kaggle or locally).
- The script will handle all steps: data transforms, loading, model setup, and training.

## 🚀 Quick Start

### Choose Your Loss Function Strategy

1. **Beginners**: Start with `dicece`
```bash
python kaggle_run.py --loss_type dicece --use_class_weights
```

2. **Standard Use**: `generalized_dice_focal`
```bash
python kaggle_run.py --loss_type generalized_dice_focal --gdl_lambda 1.0 --lambda_focal 0.5
```

3. **Competition/Research**: `adaptive_progressive_hybrid`
```bash
python kaggle_run.py --loss_type adaptive_progressive_hybrid \
  --use_adaptive_scheduling --structure_epochs 40 --boundary_epochs 70 \
  --use_warm_restarts --restart_period 30
```

### Running on Kaggle

1. Install the package (see above).
2. Run with your chosen strategy:

```python
!python ./SwinUNETRV2/kaggle_run.py --loss_type generalized_dice_focal --use_class_weights
```

## Usage

> **📖 Note**: The examples below are just a basic overview. For comprehensive usage examples, detailed parameter explanations, and advanced configurations, check:
> - **[kaggle_run.py](kaggle_run.py)** - Complete parameter definitions and example commands
> - **[docs/loss.md](docs/loss.md)** - Detailed loss function guide with usage examples
> - **[docs/pipeline.md](docs/pipeline.md)** - Complete pipeline documentation

### Advanced Configuration

SwinUNETR++ offers 30+ CLI parameters for comprehensive customization:

#### Loss Function Options (14 total):
```bash
--loss_type dice|dicece|dicefocal|generalized_dice|generalized_dice_focal|
           focal|tversky|hausdorff|hybrid_gdl_focal_tversky|hybrid_dice_hausdorff|
           adaptive_structure_boundary|adaptive_progressive_hybrid|
           adaptive_complexity_cascade|adaptive_dynamic_hybrid
```

#### Adaptive Scheduling:
```bash
--use_adaptive_scheduling  # Enable adaptive loss scheduling
--adaptive_schedule_type linear|exponential|cosine
--structure_epochs 30      # Focus on structure learning
--boundary_epochs 50       # Focus on boundary refinement
--min_loss_weight 0.1      # Minimum loss component weight
--max_loss_weight 2.0      # Maximum loss component weight
```

#### Local Minima Escape:
```bash
--use_warm_restarts        # Enable warm restarts
--restart_period 20        # Restart every N epochs
--restart_mult 1           # Period multiplier
```

#### Loss-Specific Parameters:
```bash
--focal_gamma 2.0          # Focal loss focusing parameter
--tversky_alpha 0.5        # Tversky precision/recall balance
--hausdorff_alpha 2.0      # Hausdorff distance scaling
--class_weights 3.0 1.0 5.0  # TC, WT, ET weights
```

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
from src.models.swinunetr import SwinUNETR
from src.models.pipeline import BrainTumorSegmentation

# Initialize SwinUNETR++ with adaptive losses
model = BrainTumorSegmentation(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_type='adaptive_progressive_hybrid',
    use_adaptive_scheduling=True,
    use_warm_restarts=True
)
```

## 📚 Documentation

- **[docs/loss.md](docs/loss.md)** - ⭐ Complete loss function guide with SOTA strategies
- **[docs/pipeline.md](docs/pipeline.md)** - Pipeline architecture overview
- **[docs/README.md](docs/README.md)** - Documentation index

## Advanced Features

### BraTS Optimizations
- **Class Imbalance Handling**: Specialized weights for TC/WT/ET regions
- **Multi-Objective Training**: Balance volume accuracy and boundary quality
- **Progressive Learning**: Curriculum from structure to fine details

### Local Minima Escape
- **Warm Restarts**: Periodic LR spikes to escape poor optima
- **Plateau Detection**: Automatic identification of training stagnation
- **Loss Switching**: Dynamic strategy adaptation during training

## License

MIT License