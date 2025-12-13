"""
Comprehensive evaluation script for comparing pruning methods
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import copy
from datetime import datetime

from src.utils.data_loader import load_mnist, load_fashion_mnist
from src.utils.evaluation import (
    calculate_auc, evaluate_accuracy, get_mcc_and_precision,
    get_model_size_kb, count_nonzero_params,
    plot_accuracy_comparison, plot_model_size_comparison,
    plot_metric_curves, plot_auc_comparison,
    statistical_comparison, print_auc_summary, save_results_to_file
)
from src.models.simple_cnn import SimpleCNN
from src.pruning.magnitude_pruning import MagnitudePruner


def train_single_model(model, train_loader, criterion, optimizer, num_epochs, device='cpu'):
    """Train a single model"""
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if (epoch + 1) % 5 == 0:
            acc = 100. * correct / total
            print(f"  Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
    
    return model


def evaluate_pruning_sweep(model, test_loader, prune_pcts, pruner_class, device='cpu'):
    """
    Evaluate a pruning method across different pruning percentages
    
    Returns: accuracies, model_sizes, mccs, precisions
    """
    accuracies = []
    model_sizes = []
    mccs = []
    precisions = []
    
    initial_acc = evaluate_accuracy(model, test_loader, device)
    
    for pct in prune_pcts:
        # Create a copy of the model for pruning
        pruned_model = copy.deepcopy(model)
        
        # Apply pruning
        pruner = pruner_class(threshold=pct)  # For magnitude pruning
        pruner.apply_pruning(pruned_model)
        
        # Evaluate
        acc = evaluate_accuracy(pruned_model, test_loader, device)
        size = get_model_size_kb(pruned_model)
        mcc, precision = get_mcc_and_precision(pruned_model, test_loader, device)
        
        accuracies.append(acc)
        model_sizes.append(size)
        mccs.append(mcc)
        precisions.append(precision)
        
        del pruned_model  # Free memory
    
    return accuracies, model_sizes, mccs, precisions


def cross_validation_evaluation(config, num_folds=5, prune_range=(0.0, 1.0, 0.05)):
    """
    Perform cross-validation evaluation of pruning methods
    
    Args:
        config: Configuration dictionary
        num_folds: Number of cross-validation folds
        prune_range: Tuple of (start, end, step) for pruning percentages
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Setup
    dataset_name = config.get('dataset', {}).get('name', 'mnist')
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    
    # Load data
    if dataset_name.lower() == 'mnist':
        train_loader = load_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_mnist(batch_size=batch_size, train=False, download=True)
    else:
        train_loader = load_fashion_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_fashion_mnist(batch_size=batch_size, train=False, download=True)
    
    # Create pruning percentage range
    prune_pcts = np.arange(prune_range[0], prune_range[1] + prune_range[2], prune_range[2])
    
    # Storage for results
    all_results = {
        'magnitude': {
            'accuracies': [],
            'model_sizes': [],
            'mccs': [],
            'precisions': [],
            'aucs': []
        }
    }
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Starting Cross-Validation Evaluation")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of folds: {num_folds}")
    print(f"Pruning range: {prune_range[0]:.0%} to {prune_range[1]:.0%} (step {prune_range[2]:.0%})")
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}\n")
    
    # Cross-validation loop
    for fold in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{num_folds}")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        torch.manual_seed(fold)
        np.random.seed(fold)
        
        # Initialize model
        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        print("Training model...")
        model = train_single_model(model, train_loader, criterion, optimizer, num_epochs, device)
        
        # Save trained model
        model_path = os.path.join(results_dir, f'model_fold_{fold}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate baseline
        baseline_acc = evaluate_accuracy(model, test_loader, device)
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        
        # Evaluate pruning methods
        print("\nEvaluating magnitude pruning...")
        accs, sizes, mccs, precs = evaluate_pruning_sweep(
            model, test_loader, prune_pcts, MagnitudePruner, device
        )
        
        # Normalize accuracies
        normalized_accs = [a / baseline_acc for a in accs]
        
        # Calculate AUC
        auc = calculate_auc(prune_pcts, normalized_accs)
        print(f"AUC (normalized): {auc:.4f}")
        
        # Store results
        all_results['magnitude']['accuracies'].append(accs)
        all_results['magnitude']['model_sizes'].append(sizes)
        all_results['magnitude']['mccs'].append(mccs)
        all_results['magnitude']['precisions'].append(precs)
        all_results['magnitude']['aucs'].append(auc)
    
    # Compute statistics
    print(f"\n{'='*60}")
    print("Computing Statistics Across Folds")
    print(f"{'='*60}\n")
    
    # Mean and std for each metric
    for method, results in all_results.items():
        print(f"{method.upper()} Pruning:")
        
        # Accuracies
        mean_accs = np.mean(results['accuracies'], axis=0)
        std_accs = np.std(results['accuracies'], axis=0)
        
        # Model sizes
        mean_sizes = np.mean(results['model_sizes'], axis=0)
        std_sizes = np.std(results['model_sizes'], axis=0)
        
        # MCCs
        mean_mccs = np.mean(results['mccs'], axis=0)
        std_mccs = np.std(results['mccs'], axis=0)
        
        # Precisions
        mean_precs = np.mean(results['precisions'], axis=0)
        std_precs = np.std(results['precisions'], axis=0)
        
        # AUCs
        mean_auc = np.mean(results['aucs'])
        std_auc = np.std(results['aucs'])
        
        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print()
    
    # Generate plots
    print("Generating plots...")
    
    # Accuracy plot
    acc_dict = {
        'Magnitude Pruning': np.mean(all_results['magnitude']['accuracies'], axis=0)
    }
    plot_accuracy_comparison(
        prune_pcts, acc_dict,
        title=f"Accuracy vs. Pruning Fraction ({dataset_name.upper()})",
        save_path=os.path.join(results_dir, 'accuracy_comparison.png')
    )
    
    # Model size plot
    size_dict = {
        'Magnitude Pruning': np.mean(all_results['magnitude']['model_sizes'], axis=0)
    }
    plot_model_size_comparison(
        prune_pcts, size_dict,
        title=f"Model Size vs. Pruning Fraction ({dataset_name.upper()})",
        save_path=os.path.join(results_dir, 'size_comparison.png')
    )
    
    # MCC plot
    plot_metric_curves(
        {'Magnitude': all_results['magnitude']['mccs']},
        "Matthews Correlation Coefficient (MCC)",
        prune_pcts,
        save_path=os.path.join(results_dir, 'mcc_comparison.png')
    )
    
    # Precision plot
    plot_metric_curves(
        {'Magnitude': all_results['magnitude']['precisions']},
        "Precision",
        prune_pcts,
        save_path=os.path.join(results_dir, 'precision_comparison.png')
    )
    
    # AUC comparison
    plot_auc_comparison(
        {'Magnitude': all_results['magnitude']['aucs']},
        title="Mean AUC Comparison (Cross-Validation)",
        save_path=os.path.join(results_dir, 'auc_comparison.png')
    )
    
    # Save numerical results
    summary = {
        'config': config,
        'num_folds': num_folds,
        'prune_range': prune_range,
        'results': all_results
    }
    
    save_results_to_file(summary, os.path.join(results_dir, 'results_summary.txt'))
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to {results_dir}")
    print(f"{'='*60}\n")
    
    return all_results


def main():
    """Main evaluation function"""
    # Load config
    config_path = os.path.join('src', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run evaluation
    # Full range: 0-100%
    print("Running full range evaluation (0-100%)...")
    results_full = cross_validation_evaluation(
        config,
        num_folds=5,
        prune_range=(0.0, 1.0, 0.05)  # 0%, 5%, 10%, ..., 100%
    )
    
    # High pruning range: 80-100%
    print("\nRunning high pruning range evaluation (80-100%)...")
    results_high = cross_validation_evaluation(
        config,
        num_folds=5,
        prune_range=(0.80, 1.0, 0.02)  # 80%, 82%, 84%, ..., 100%
    )


if __name__ == '__main__':
    main()