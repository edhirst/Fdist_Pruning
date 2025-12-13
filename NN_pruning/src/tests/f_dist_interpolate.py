import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.pruning.sqrt_averaged_magnitude_fim_pruner import SqrtAveragedMagnitudeFIMPruner

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

def get_dummy_dataloader(num_samples=20, input_dim=10, num_classes=10, batch_size=5):
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)

def test_sqrt_averaged_magnitude_fim_pruner_increases_zeros():
    model = DummyModel()
    config = {'prune_amount': 0.5, 'fim_samples': 10, 'global_pruning': True}
    pruner = SqrtAveragedMagnitudeFIMPruner(config)
    dataloader = get_dummy_dataloader()
    zeros_before = (model.fc.weight == 0).sum().item()
    pruned_model = pruner.prune(model, dataloader=dataloader)
    zeros_after = (pruned_model.fc.weight == 0).sum().item()
    assert zeros_after > zeros_before, "Sqrt Averaged Magnitude x FIM pruning did not increase the number of zero weights!"