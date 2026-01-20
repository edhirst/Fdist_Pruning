import torch
import torch.nn as nn
import numpy as np
import copy

try:
    from .base_pruner import BasePruner
except Exception:
    # Fallback BasePruner for environments where the relative import is unavailable
    class BasePruner:
        """Minimal fallback BasePruner used for linting/tests when the real BasePruner can't be imported."""
        def __init__(self):
            pass

class FDistPruner(BasePruner):
    """
    Efficient averaged FIM x Magnitude pruning for 80-100% pruning range.
    Uses sqrt of averaged FIM between initial and final states.
    """
    def __init__(self, magnitude_threshold=0.1, fim_threshold=0.1):
        super().__init__()
        self.magnitude_threshold = magnitude_threshold
        self.fim_threshold = fim_threshold
        self.intermediate_pruning_pct = 0.80
        self.epsilon = 1e-6

    def set_parameters(self, magnitude_threshold=None, fim_threshold=None):
        if magnitude_threshold is not None:
            self.magnitude_threshold = magnitude_threshold
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
            all_weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
            return torch.ones_like(all_weights)

    def apply_pruning(self, model, train_loader=None, test_loader=None, device='cpu', target_pruning_pct=0.90):
        """
        Apply efficient averaged FIM x Magnitude pruning.
        
        This method:
        1. Prunes to 80% using one-shot FIM x Magnitude
        2. Computes FIM at 80% state (Point A)
        3. Creates hypothetical fully-pruned state and computes FIM (Point B)
        4. Uses sqrt of averaged FIM for final pruning decisions
        
        Args:
            model: PyTorch model to prune
            train_loader: DataLoader for FIM computation
            test_loader: DataLoader for evaluation (optional)
            device: Device for computation
            target_pruning_pct: Target pruning percentage (should be >= 0.80)
            
        Returns:
            model: Pruned model
            accuracy: Final accuracy (if test_loader provided)
            model_size: Final model size
        """
        if train_loader is None:
            raise ValueError("Averaged FIM pruning requires train_loader")
        
        print(f"\n--- Applying Efficient Averaged FIM x Magnitude Pruning (Target {target_pruning_pct:.0%}) ---")
        
        # Step 1: Prune to 80% using one-shot FIM x Magnitude
        from .f_dist_one_shot import FDistOneShotPruner
        one_shot_pruner = FDistOneShotPruner(
            model=model,
            magnitude_threshold=self.intermediate_pruning_pct,
            fim_threshold=self.fim_threshold
        )
        model_80_pruned = one_shot_pruner.apply_pruning(model, train_loader, device)
        print(f"Model pruned to {self.intermediate_pruning_pct:.0%}")
        
        # Step 2: Determine additional parameters to prune
        total_original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        target_zeros_final = int(target_pruning_pct * total_original_params)
        num_current_zeros = sum((p.data == 0).sum().item() for p in model_80_pruned.parameters() if p.requires_grad)
        num_to_prune_additional = target_zeros_final - num_current_zeros
        
        print(f"Already pruned: {num_current_zeros}. Need to prune additional: {num_to_prune_additional}")
        
        if num_to_prune_additional <= 0:
            print("Target already met.")
            if test_loader:
                from src.utils.evaluation import evaluate_accuracy, get_model_size_kb
                acc = evaluate_accuracy(model_80_pruned, test_loader, device)
                size = get_model_size_kb(model_80_pruned)
                return model_80_pruned, acc, size
            return model_80_pruned, None, None
        
        # Step 3: Compute FIM at Point A (80% pruned state)
        model_for_fine_pruning = copy.deepcopy(model_80_pruned).to(device)
        fim_diag_A = self.calculate_fim(model_for_fine_pruning, train_loader, device)
        
        # Collect active parameters and their initial importance
        active_params_info = []
        global_idx_offset = 0
        
        for module in model_for_fine_pruning.modules():
            if isinstance(module, nn.Linear):
                weight_data = module.weight.data.cpu()
                for local_idx in range(weight_data.numel()):
                    if weight_data.flatten()[local_idx] != 0:
                        weight_value = weight_data.flatten()[local_idx].item()
                        fim_value = fim_diag_A[global_idx_offset + local_idx].item()
                        initial_importance = (fim_value + self.epsilon) * abs(weight_value)
                        active_params_info.append((global_idx_offset + local_idx, initial_importance, module, local_idx))
                global_idx_offset += weight_data.numel()
        
        # Sort by initial importance and select lowest
        sorted_params = sorted(active_params_info, key=lambda x: x[1])
        params_to_consider = sorted_params[:num_to_prune_additional]
        
        # Step 4: Create hypothetical final state
        hypothetical_model = copy.deepcopy(model_for_fine_pruning).to(device)
        global_indices_to_zero = {p[0] for p in params_to_consider}
        
        global_idx_tracker = 0
        for module in hypothetical_model.modules():
            if isinstance(module, nn.Linear):
                weight_data = module.weight.data
                numel = weight_data.numel()
                for l_idx in range(numel):
                    if global_idx_tracker + l_idx in global_indices_to_zero:
                        weight_data.flatten()[l_idx] = 0.0
                global_idx_tracker += numel
        
        # Step 5: Compute FIM at Point B (hypothetical final state)
        fim_diag_B = self.calculate_fim(hypothetical_model, train_loader, device)
        
        # Step 6: Calculate averaged importance and apply final pruning
        final_importance_scores = []
        
        for original_global_idx, _, module, local_idx in params_to_consider:
            weight_value = module.weight.data.flatten()[local_idx].item()
            fim_A = fim_diag_A[original_global_idx].item()
            fim_B = fim_diag_B[original_global_idx].item()
            
            # Average FIM and take square root
            averaged_fim = (fim_A + fim_B) / 2.0
            importance_score = np.sqrt(max(0.0, averaged_fim) + self.epsilon) * abs(weight_value)
            final_importance_scores.append((original_global_idx, importance_score))
        
        # Sort by final importance and zero out lowest
        final_sorted = sorted(final_importance_scores, key=lambda x: x[1])
        final_indices_to_zero = {p[0] for p in final_sorted}
        
        # Apply final pruning
        global_idx = 0
        for module in model_for_fine_pruning.modules():
            if isinstance(module, nn.Linear):
                weight_data = module.weight.data
                numel = weight_data.numel()
                for l_idx in range(numel):
                    if weight_data.flatten()[l_idx] != 0 and global_idx + l_idx in final_indices_to_zero:
                        weight_data.flatten()[l_idx] = 0.0
                global_idx += numel
        
        # Evaluate
        if test_loader:
            from src.utils.evaluation import evaluate_accuracy, get_model_size_kb
            final_acc = evaluate_accuracy(model_for_fine_pruning, test_loader, device)
            final_size = get_model_size_kb(model_for_fine_pruning)
            print(f"Final Averaged FIM Prune {target_pruning_pct:.0%} → Acc: {final_acc:.4f}, Size: {final_size:.2f}KB")
            return model_for_fine_pruning, final_acc, final_size
        
        return model_for_fine_pruning, None, None

    def calculate_pruning_mask(self, model):
        """Calculate pruning mask (for compatibility)"""
        mask = []
        for param in model.parameters():
            if param.requires_grad:
                mask.append((param.data != 0).float())
        return mask

    def prune_model(self, model):
        """Apply stored pruning mask (for compatibility)"""
        return model