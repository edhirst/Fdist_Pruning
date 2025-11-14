import torch
import torch.nn as nn

try:
    from .base_pruner import BasePruner
except Exception:
    # Fallback BasePruner for environments where the relative import is unavailable
    class BasePruner:
        """Minimal fallback BasePruner used for linting/tests when the real BasePruner can't be imported."""
        def __init__(self):
            pass
import torch 

class MagnitudePruner(BasePruner):
    """
    Prunes weights based on their absolute magnitude.
    Removes weights with the lowest absolute values.
    """
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold

    def set_parameters(self, threshold):
        self.threshold = threshold

    def apply_pruning(self, model):
        """
        Prunes weights globally based on magnitude threshold.
        
        Args:
            model: PyTorch model to prune
            
        Returns:
            model: Pruned model
        """
        # Collect all trainable weights
        all_weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
        
        # Calculate magnitude scores
        magnitude_scores = torch.abs(all_weights)
        
        # Determine threshold for pruning
        threshold_value = torch.quantile(magnitude_scores, self.threshold)
        
        # Create global mask
        mask_global = (magnitude_scores >= threshold_value).float()
        
        # Apply mask to each layer
        offset = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param_mask = mask_global[offset:offset + numel].view_as(param)
                param.data *= param_mask
                offset += numel
        
        # Count pruned parameters
        total_params = all_weights.numel()
        pruned_params = (mask_global == 0).sum().item()
        pruning_rate = 100.0 * pruned_params / total_params
        
        print(f"Magnitude Pruning: Pruned {pruned_params}/{total_params} parameters ({pruning_rate:.2f}%)")
        
        return model

    def __str__(self):
        return f"MagnitudePruner(threshold={self.threshold})"

