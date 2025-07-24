def _get_progressive_weights(self):
        """Get progressive weights that add complexity over time"""
        if not self.use_adaptive_scheduling or self.current_epoch < self.schedule_start_epoch:
            return 1.0, 0.0, 0.0
        
        epoch = self.current_epoch - self.schedule_start_epoch
        
        # Phase 1: Structure learning (Dice dominant)
        if epoch < self.structure_epochs:
            dice_weight = self.max_loss_weight
            focal_weight = self.min_loss_weight
            hausdorff_weight = 0.0
        # Phase 2: Add boundary refinement (Focal)
        elif epoch < self.boundary_epochs:
            progress = (epoch - self.structure_epochs) / (self.boundary_epochs - self.structure_epochs)
            dice_weight = self.max_loss_weight * (1.0 - 0.3 * progress)
            focal_weight = self.min_loss_weight + (self.max_loss_weight - self.min_loss_weight) * progress
            hausdorff_weight = 0.0
        # Phase 3: Add fine boundary details (Hausdorff)
        else:
            progress = min((epoch - self.boundary_epochs) / 20, 1.0)  # 20 epochs to ramp up
            dice_weight = self.max_loss_weight * 0.7
            focal_weight = self.max_loss_weight * 0.8
            hausdorff_weight = self.min_loss_weight + (self.max_loss_weight * 0.5 - self.min_loss_weight) * progress
            
        return dice_weight, focal_weight, hausdorff_weight