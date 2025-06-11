# SwinUNETR V2

A PyTorch implementation of SwinUNETR V2 for medical image segmentation.

## Installation

### On Kaggle

Simply install the package from GitHub:

```python
!pip install git+https://github.com/yourusername/swinunetrv2.git
```

This will automatically install all required dependencies including:
- MONAI with nibabel, tqdm, and einops
- Matplotlib
- Einops
- And other required packages

### Local Installation

```bash
pip install -e .
```

## Project Structure

```
swinunetrv2/
├── src/
│   └── swinunetrv2/
│       ├── models/         # Model architecture
│       ├── data/          # Data loading and augmentation
│       ├── utils/         # Utility functions
│       └── configs/       # Configuration files
├── tests/                 # Test files
├── setup.py              # Package setup file
└── requirements.txt      # Project dependencies
```

## Execution Flow

The code follows this sequence:
1. `convert_labels.py`: Convert and preprocess the input labels
2. `augmentations.py`: Set up data augmentation transforms
3. `dataloader.py`: Create data loaders for training and validation
4. `architecture.py`: Initialize the SwinUNETR model
5. `trainer.py`: Train the model

## Usage

### Running on Kaggle

1. Install the package:
```python
!pip install git+https://github.com/yourusername/swinunetrv2.git
```

2. Run the training:
```python
from swinunetrv2.main import main
import argparse

args = argparse.Namespace(
    input_dir='/kaggle/input/brats2023-part-1',
    output_dir='/kaggle/working/processed_data',
    batch_size=8,
    img_size=96,
    in_channels=4,  # 4 modalities for BraTS
    out_channels=3,  # 3 tumor regions for BraTS
    feature_size=12,
    learning_rate=1e-4,
    epochs=100,
    device='cuda'
)

main(args)
```

### Running Locally

You can run the entire pipeline using the main script:

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
from swinunetrv2.models.architecture import SwinUNETR
from swinunetrv2.trainer import Trainer

# Initialize model
model = SwinUNETR(...)

# Train model
trainer = Trainer(model, ...)
trainer.train()
```

## License

MIT License 