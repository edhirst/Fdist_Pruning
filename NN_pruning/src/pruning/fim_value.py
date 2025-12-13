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

class fim_value(BasePruner):
    """
    Prunes weights based on Fisher Information Matrix (FIM).
    Requires nngeometry library for FIM computation.
    """
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.threshold = self.parameters.get('pruning_threshold', 0.1)
        self.epsilon = 1e-6

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.threshold = self.parameters.get('pruning_threshold', 0.1)

    def calculate_fim(self, model, train_loader, device='cpu'):
        """
        Calculate Fisher Information Matrix diagonal for the model.
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            device: Device to run computation on
            
        Returns:
            fim_diag: Diagonal of FIM as a flat tensor
        """
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
            fim_diag = fim_obj.get_diag().cpu()
            print(f"FIM diagonal computed: shape {fim_diag.shape}")
            return fim_diag
            
        except ImportError:
            print("Warning: nngeometry not installed. Using uniform FIM (all ones).")
            all_weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
            return torch.ones_like(all_weights)

    def apply_pruning(self, model, train_loader=None, device='cpu'):
        """
        Apply FIM-based pruning to the model.
        
        Args:
            model: PyTorch model to prune
            train_loader: DataLoader for FIM computation
            device: Device for computation
            
        Returns:
            model: Pruned model
        """
        if train_loader is None:
            raise ValueError("FIM pruning requires train_loader for FIM computation")
        
        # Calculate FIM diagonal
        fim_diag = self.calculate_fim(model, train_loader, device)
        
        # Determine threshold for pruning
        threshold_value = torch.quantile(fim_diag, self.threshold)
        
        # Create global mask based on FIM values
        mask_global = (fim_diag >= threshold_value).float()
        
        # Apply mask to each layer
        offset = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param_mask = mask_global[offset:offset + numel].view_as(param)
                param.data *= param_mask
                offset += numel
        
        # Count pruned parameters
        total_params = fim_diag.numel()
        pruned_params = (mask_global == 0).sum().item()
        pruning_rate = 100.0 * pruned_params / total_params
        
        print(f"FIM Pruning: Pruned {pruned_params}/{total_params} parameters ({pruning_rate:.2f}%)")
        
        return model
