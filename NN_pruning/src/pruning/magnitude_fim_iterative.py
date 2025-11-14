import torch
import torch.nn as nn
import copy

try:
    from .base_pruner import BasePruner
except Exception:
    # Fallback BasePruner for environments where the relative import is unavailable
    class BasePruner:
        """Minimal fallback BasePruner used for linting/tests when the real BasePruner can't be imported."""
        def __init__(self):
            pass

class MagnitudeFIMIterativePruner(BasePruner):
    """
    Iterative pruning that recomputes FIM at each step.
    Only considers non-zero (active) weights for importance calculation.
    """
    def __init__(self, model=None, params=None):
        super().__init__()
        self.model = model
        params = params or {}
        self.threshold = params.get('pruning_threshold', 0.1)
        self.iterations = params.get('iterations', 10)
        self.epsilon = 1e-6

    def set_parameters(self, params):
        self.threshold = params.get('pruning_threshold', self.threshold)
        self.iterations = params.get('iterations', self.iterations)

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
            all_weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
            return torch.ones_like(all_weights)

    def apply_pruning(self, model=None, train_loader=None, test_loader=None, device='cpu', prune_pcts=None):
        """
        Apply iterative FIM x Magnitude pruning.
        
        Args:
            model: PyTorch model to prune
            train_loader: DataLoader for FIM computation
            test_loader: DataLoader for evaluation (optional)
            device: Device for computation
            prune_pcts: List of cumulative pruning percentages
            
        Returns:
            accuracies: List of accuracies at each pruning level
            model_sizes: List of model sizes at each pruning level
        """
        if model is None:
            model = self.model
        if train_loader is None:
            raise ValueError("Iterative pruning requires train_loader")
        
        if prune_pcts is None:
            # Default: prune from 0% to threshold in iterations steps
            import numpy as np
            prune_pcts = np.linspace(0, self.threshold, self.iterations)
        
        model_iterative = copy.deepcopy(model).to(device)
        total_original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        accuracies = []
        model_sizes = []
        
        for i, target_pct in enumerate(prune_pcts):
            target_zeros = int(target_pct * total_original_params)
            num_current_zeros = sum((p.data == 0).sum().item() for p in model_iterative.parameters() if p.requires_grad)
            num_to_prune_this_step = target_zeros - num_current_zeros
            
            if num_to_prune_this_step > 0:
                # Recompute FIM on current model state
                current_fim_diag = self.calculate_fim(model_iterative, train_loader, device)
                
                # Collect active (non-zero) weights and their indices
                active_weight_values = []
                active_fim_values = []
                active_global_indices = []
                
                global_idx_offset = 0
                for module in model_iterative.modules():
                    if isinstance(module, nn.Linear):
                        weight_data = module.weight.data.cpu()
                        for local_idx in range(weight_data.numel()):
                            if weight_data.flatten()[local_idx] != 0:
                                active_weight_values.append(weight_data.flatten()[local_idx])
                                active_fim_values.append(current_fim_diag[global_idx_offset + local_idx])
                                active_global_indices.append(global_idx_offset + local_idx)
                        global_idx_offset += weight_data.numel()
                
                if active_weight_values:
                    active_weight_values_tensor = torch.stack(active_weight_values)
                    active_fim_values_tensor = torch.stack(active_fim_values)
                    active_global_indices_tensor = torch.tensor(active_global_indices)
                    
                    # Calculate combined importance for active parameters
                    combined_importance = (active_fim_values_tensor + self.epsilon) * torch.abs(active_weight_values_tensor)
                    
                    # Select lowest importance parameters to prune
                    sorted_indices = torch.argsort(combined_importance)
                    num_to_prune_actual = min(num_to_prune_this_step, combined_importance.numel())
                    indices_to_prune = sorted_indices[:num_to_prune_actual]
                    global_indices_to_zero = active_global_indices_tensor[indices_to_prune]
                    
                    # Zero out selected weights
                    current_param_offset = 0
                    for module in model_iterative.modules():
                        if isinstance(module, nn.Linear):
                            weight_data = module.weight.data
                            numel = weight_data.numel()
                            for global_idx in global_indices_to_zero:
                                if current_param_offset <= global_idx < current_param_offset + numel:
                                    local_idx = global_idx - current_param_offset
                                    weight_data.flatten()[local_idx] = 0.0
                            current_param_offset += numel
            
            # Evaluate if test_loader provided
            if test_loader is not None:
                from src.utils.evaluation import evaluate_accuracy, get_model_size_kb
                acc = evaluate_accuracy(model_iterative, test_loader, device)
                size = get_model_size_kb(model_iterative)
                accuracies.append(acc)
                model_sizes.append(size)
                print(f"Iteration {i+1}/{len(prune_pcts)} - Pruned {target_pct:.0%}: Acc={acc:.4f}, Size={size:.2f}KB")
        
        # Update the original model
        if model is not None:
            model.load_state_dict(model_iterative.state_dict())
        
        return accuracies, model_sizes
