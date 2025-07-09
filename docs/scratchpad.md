# Scratchpad

## TODOs
- [ ] Add tests for data pipeline and model training
- [ ] Improve CLI argument validation in `main.py`
- [ ] Add more data augmentations (elastic, gamma, etc.)
- [ ] Refactor visualization to support more modalities
- [ ] Add support for BraTS 2024 if format changes
- [ ] Document all public functions/classes
- [ ] Add example notebooks for inference and visualization

## Experimental Notes
- Current training uses only 20% of training data for speed; revert for full runs.
- Class weights for Dice/Focal loss are set for BraTS imbalance (can tune).
- SwinUNETR-V2 uses residual conv blocks at each Swin stage (see `use_v2`).
- Sliding window inference overlap is 0.6 for validation.
- Visualization currently assumes T1c as first channel; generalize for multi-modal.

## Ideas
- Add self-supervised pretraining for SwinUNETR backbone
- Integrate more advanced logging (TensorBoard, custom callbacks)
- Add support for test-time augmentation
- Modularize config (YAML/JSON) for reproducibility 