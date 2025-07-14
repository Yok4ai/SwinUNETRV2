# Cursor Project Rules and Guidelines

## Project Overview
This project implements an advanced SwinUNETR V2 pipeline for brain tumor segmentation with comprehensive loss function optimization and local minima escape strategies. Built on MONAI and PyTorch Lightning with extensive customization options.

## Key Features
- **14 Loss Functions**: Basic, hybrid, and adaptive loss strategies
- **Adaptive Scheduling**: Dynamic loss weight transitions for curriculum learning
- **Local Minima Escape**: Warm restarts and plateau detection mechanisms
- **BraTS Optimization**: Specialized for brain tumor segmentation challenges
- **Comprehensive CLI**: 30+ configurable parameters for research and competition use

## Folder Structure
- `src/data/`: Data loading, augmentation, and label conversion (supports BRATS 2021/2023 conventions)
- `src/models/`: Model architecture (SwinUNETR+), training pipeline with adaptive loss scheduling
- `src/utils/`: Visualization and utility functions
- `docs/`: Comprehensive documentation including loss function guide and SOTA strategies
- `kaggle_run.py`: Main CLI entry point with full parameter control
- `main.py`: Core training orchestration

## Code Style
- Use PEP8 for Python code.
- Use docstrings for all functions and classes.
- Use type hints where possible.
- Prefer explicit over implicit imports.
- Use snake_case for functions and variables, PascalCase for classes.
- Keep functions short and focused.
- Avoid hardcoding values; use arguments/configs.

## Collaboration Rules
- All new features must include tests or usage examples.
- Document all public functions and classes.
- Use clear, descriptive commit messages.
- Run linting and tests before pushing.
- Discuss major changes in issues or PRs before implementation.

## Project Requirements
- Python 3.8+
- MONAI, PyTorch, PyTorch Lightning, scikit-learn, matplotlib, wandb
- Data must be in BraTS format with a `dataset.json` file.

## Data Pipeline
- Data is loaded and split (80/20 train/val) from `dataset.json`.
- Augmentations and label conversion are handled in `src/data/`.
- Training and validation transforms are modular and configurable.
- **Label conversion now supports a `dataset` argument to select between BRATS 2021 and BRATS 2023 label conventions.**

## Model & Training
- SwinUNETR-V2 is the core model, with configurable depths, heads, and feature sizes.
- Training uses PyTorch Lightning, with early stopping, checkpointing, and wandb logging.
- Loss: Hybrid DiceCE + Focal, with class weighting for imbalance.

## Utilities
- Visualization tools for predictions and training curves are in `src/utils/visualization.py`.

## Changes & Extensions
- All changes must be reflected in the docs (`architecture.md`, `cursor.md`, `scratchpad.md`).
- Major refactors or new modules should be described in `architecture.md`.
- Use `scratchpad.md` for brainstorming, TODOs, and experimental notes.
- **If you add new dataset options or label conventions, document them here and in `architecture.md`.** 