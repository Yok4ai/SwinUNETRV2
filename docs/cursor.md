# Cursor Project Rules and Guidelines

## Project Overview
This project implements a SwinUNETR-V2-based pipeline for brain tumor segmentation using MONAI and PyTorch Lightning. The codebase is modular, with clear separation between data processing, model architecture, training, and utilities.

## Folder Structure
- `src/data/`: Data loading, augmentation, and label conversion.
- `src/models/`: Model architecture (SwinUNETR), training pipeline, and PyTorch Lightning integration.
- `src/utils/`: Visualization and utility functions.
- `main.py`: Entry point for training.

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