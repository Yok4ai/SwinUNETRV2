def _get_progressive_weights(self):
    """Get progressive weights that add complexity over time"""
    if not self.use_adaptive_scheduling or self.current_epoch < self.schedule_start_epoch:
        return 1.0, 0.0, 0.0
    
    epoch = self.current_epoch - self.schedule_start_epoch
    
    # Phase 1: Structure learning (Dice dominant) - gradual introduction of focal
    if epoch < self.structure_epochs:
        # Very gradual introduction of focal loss to avoid sudden drops
        progress = epoch / self.structure_epochs
        dice_weight = self.max_loss_weight  # Keep Dice strong
        focal_weight = self.min_loss_weight * (1.0 + 2.0 * progress)  # Slowly increase focal from 0.1 to 0.3
        hausdorff_weight = 0.0
    # Phase 2: Add boundary refinement (Focal) - more conservative transition
    elif epoch < self.boundary_epochs:
        progress = (epoch - self.structure_epochs) / (self.boundary_epochs - self.structure_epochs)
        # Much more conservative reduction of dice weight
        dice_weight = self.max_loss_weight * (1.0 - 0.15 * progress)  # Reduce by only 15% instead of 30%
        # More gradual increase of focal weight
        focal_weight = self.min_loss_weight * 3.0 + (self.max_loss_weight * 0.6 - self.min_loss_weight * 3.0) * progress
        hausdorff_weight = 0.0
    # Phase 3: Add fine boundary details (Hausdorff)
    else:
        progress = min((epoch - self.boundary_epochs) / 20, 1.0)  # 20 epochs to ramp up
        dice_weight = self.max_loss_weight * 0.85  # Keep dice stronger
        focal_weight = self.max_loss_weight * 0.6   # Reduce focal slightly
        hausdorff_weight = self.min_loss_weight + (self.max_loss_weight * 0.3 - self.min_loss_weight) * progress  # Smaller hausdorff contribution
        
    return dice_weight, focal_weight, hausdorff_weight