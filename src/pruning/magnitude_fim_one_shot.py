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

class MagnitudeFIMOneShotPruner(BasePruner):
    """
    One-shot pruning combining magnitude and FIM scores.
    Combined score = FIM_diagonal * |weight|
    """
    def __init__(self, model=None, magnitude_threshold=0.1, fim_threshold=0.1):
        super().__init__()
        self.model = model
        self.magnitude_threshold = magnitude_threshold
        self.fim_threshold = fim_threshold
        self.threshold = magnitude_threshold  # Use magnitude threshold as primary
        self.epsilon = 1e-6

    def set_parameters(self, magnitude_threshold=None, fim_threshold=None):
        if magnitude_threshold is not None:
            self.magnitude_threshold = magnitude_threshold
            self.threshold = magnitude_threshold
        if fim_threshold is not None:
            self.fim_threshold = fim_threshold

    def calculate_fim(self, model, train_loader, device='cpu'):
        """Calculate FIM diagonal"""
        try:
            from nngeometry import FIM
            from nngeometry.object import PMatKFAC
            
            fim_obj = FIM(
                model=model,
                loader=train_loader,
                representation=PMatKFAC,
                variant='classif_logits',
                device=device
            )
            return fim_obj.get_diag().cpu()
            
        except ImportError:
            print("Warning: nngeometry not installed. Using uniform FIM.")
            all_weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
            return torch.ones_like(all_weights)

    def apply_pruning(self, model=None, train_loader=None, device='cpu'):
        """
        Apply one-shot FIM x Magnitude pruning.
        
        Args:
            model: PyTorch model to prune
            train_loader: DataLoader for FIM computation
            device: Device for computation
            
        Returns:
            model: Pruned model
        """
        if model is None:
            model = self.model
        if train_loader is None:
            raise ValueError("One-shot FIM x Magnitude pruning requires train_loader")
        
        # Get all trainable weights
        all_weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad]).cpu()
        
        # Calculate FIM diagonal
        fim_diag = self.calculate_fim(model, train_loader, device)
        
        # Calculate combined importance score: (FIM + epsilon) * |weight|
        combined_scores = (fim_diag + self.epsilon) * torch.abs(all_weights)
        
        # Determine threshold for pruning
        threshold_value = torch.quantile(combined_scores, self.threshold)
        
        # Create global mask
        mask_global = (combined_scores >= threshold_value).float()
        
        # Apply mask to each layer
        offset = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param_mask = mask_global[offset:offset + numel].view_as(param).to(param.device)
                param.data *= param_mask
                offset += numel
        
        # Count pruned parameters
        total_params = all_weights.numel()
        pruned_params = (mask_global == 0).sum().item()
        pruning_rate = 100.0 * pruned_params / total_params
        
        print(f"FIM x Magnitude (One-Shot): Pruned {pruned_params}/{total_params} parameters ({pruning_rate:.2f}%)")
        
        return model
