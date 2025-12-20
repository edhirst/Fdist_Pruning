import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
from src.utils.data_loader import load_mnist, load_fashion_mnist
from .models.simple_cnn import SimpleCNN
from .models.simple_nn import SimpleNN  
from .pruning.magnitude_pruning import MagnitudePruner
from .pruning.fim_pruning import FIMPruner
from .pruning.magnitude_fim_one_shot import MagnitudeFIMOneShotPruner
from .pruning.magnitude_fim_iterative import MagnitudeFIMIterativePruner
from .pruning.sqrt_averaged_magnitude_fim import SqrtAveragedMagnitudeFIMPruner

def train_epoch(model, train_loader, criterion, optimizer):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


from torch.utils.data import DataLoader, Subset
import torch


def build_fim_dataloader(train_loader, subset_size=0, seed=42, shuffle_subset=True, max_batch_size=512):
    """
    Build a DataLoader for FIM computation with a fixed batch size policy:

      - If subset_size <= 0 or subset_size >= len(dataset): use full dataset.
      - If subset_size <= max_batch_size: batch_size = subset_size (single batch).
      - If subset_size >  max_batch_size: batch_size = max_batch_size.

    Args:
        train_loader: the original training DataLoader (we reuse its dataset)
        subset_size: number of samples used for FIM estimation (0 means full dataset)
        seed: random seed for subset sampling
        shuffle_subset: whether to randomly sample subset indices
        max_batch_size: cap for FIM batch size (default 512)

    Returns:
        fim_loader: DataLoader for FIM computation
    """
    dataset = train_loader.dataset
    n = len(dataset)

    subset_size = int(subset_size)

    # Select dataset (full or subset)
    if subset_size <= 0 or subset_size >= n:
        fim_dataset = dataset
        effective_subset_size = n
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))

        if shuffle_subset:
            indices = torch.randperm(n, generator=g)[:subset_size].tolist()
        else:
            indices = list(range(subset_size))

        fim_dataset = Subset(dataset, indices)
        effective_subset_size = subset_size

    # Apply batch size policy
    if effective_subset_size <= max_batch_size:
        fim_batch_size = effective_subset_size
    else:
        fim_batch_size = max_batch_size

    return DataLoader(fim_dataset, batch_size=fim_batch_size, shuffle=False)



def train_model(config):
    """Main training function that takes config as input"""
    
    # Load dataset based on config
    dataset_name = config.get('dataset', {}).get('name', 'mnist')
    batch_size = config['training']['batch_size']
    
    if dataset_name.lower() == 'mnist':
        train_loader = load_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_mnist(batch_size=batch_size, train=False, download=True)
    elif dataset_name.lower() == 'fashion_mnist':
        train_loader = load_fashion_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_fashion_mnist(batch_size=batch_size, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    

    # Initialize model
    model_cfg = config.get("model", {}) or {}
    common_cfg = model_cfg.get("common", {}) or {}

    model_type = common_cfg.get("model_type", "NN")
    num_classes = common_cfg.get("num_classes", 10)

    if model_type == "NN":
        nn_cfg = model_cfg.get("NN", {}) or {}
        hidden_size = nn_cfg.get("hidden_size", 32)
        hidden_layers = nn_cfg.get("hidden_layers", 2)

        model = SimpleNN(hidden_size = hidden_size, hidden_layers = hidden_layers, num_classes=num_classes)
        print(f"\nInitialized Model: Simple Fully Connected Neuron Network")
        print(f"{hidden_layers} hidden layer(s), each with {hidden_size} neurons.")


    elif model_type == "cnn":
        cnn_cfg = model_cfg.get("cnn", {}) or {}
        # If your SimpleCNN accepts extra args, read from cnn_cfg here.
        # For now, only num_classes:
        model = SimpleCNN(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model_type: {common_cfg.get('model_type')}. "
            "Supported: 'SimpleNN', 'CNN'."
        )














    # Setup training
    criterion = nn.CrossEntropyLoss()
    learning_rate = config['training']['learning_rate']
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    pruning_cfg = config.get("pruning", {}) or {}
    fim_subset_size = int(pruning_cfg.get("fim_subset_size", 0))
    fim_subset_seed = int(pruning_cfg.get("fim_subset_seed", 42))
    fim_subset_shuffle = bool(pruning_cfg.get("fim_subset_shuffle", True))

    fim_loader = build_fim_dataloader(
        train_loader,
        subset_size = fim_subset_size,
        seed = fim_subset_seed,
        shuffle_subset = fim_subset_shuffle,
        max_batch_size = 512,
    )

    # Initialize pruning scheme if enabled
    if config.get('pruning', {}).get('enable_pruning', False):
        pruning_scheme = config['pruning'].get('pruning_scheme', 'magnitude')
        pruning_params = config['pruning']
        
        pruner = None
        if pruning_scheme == 'magnitude':
            threshold = pruning_params.get('pruning_threshold', 0.1)
            pruner = MagnitudePruner(threshold=threshold)
        elif pruning_scheme == 'fim':
            pruner = FIMPruner(parameters=pruning_params)
        elif pruning_scheme == 'magnitude_fim_one_shot':
            mag_thresh = pruning_params.get('pruning_threshold', 0.1)
            fim_thresh = pruning_params.get('fim_threshold', 0.1)
            pruner = MagnitudeFIMOneShotPruner(model, mag_thresh, fim_thresh)
        elif pruning_scheme == 'magnitude_fim_iterative':
            pruner = MagnitudeFIMIterativePruner(model, pruning_params)
        elif pruning_scheme == 'sqrt_averaged_magnitude_fim':
            mag_thresh = pruning_params.get('pruning_threshold', 0.1)
            fim_thresh = pruning_params.get('fim_threshold', 0.1)
            pruner = SqrtAveragedMagnitudeFIMPruner(mag_thresh, fim_thresh)
        
    
    # Train the model
    num_epochs = config['training']['num_epochs']
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
    
    print("-" * 50)
    print("Pretrained completed!")


    # Prune the model
    if pruner:
        print(f"Applying {pruning_scheme} pruning...")
        pruner.apply_pruning(model, fim_loader)
    
    # Save model if path specified
    model_save_path = config.get('paths', {}).get('model_save_path')
    if model_save_path:
        import os
        os.makedirs(model_save_path, exist_ok=True)
        save_file = os.path.join(model_save_path, 'trained_model.pth')
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to {save_file}")
    
    return model