# Pruning Evaluation Guide

This guide explains how to use the evaluation features in this project.

## Quick Start

### 1. Basic Training
```bash
# Train a single model with pruning
python -m src.main
```

### 2. Comprehensive Evaluation
```bash
# Run full cross-validation evaluation
python -m src.evaluate_pruning
```

## Features

### Metrics Computed

1. **Accuracy**: Classification accuracy on test set
2. **Model Size**: Size of model in KB
3. **MCC (Matthews Correlation Coefficient)**: Balanced metric for classification
4. **Precision**: Macro-averaged precision across classes
5. **AUC**: Area under the accuracy vs. pruning curve

### Evaluation Modes

#### Full Range Evaluation (0-100%)
Evaluates pruning across the entire range from 0% to 100% pruning.
```python
prune_range=(0.0, 1.0, 0.05)  # 0%, 5%, 10%, ..., 100%
```

#### High Pruning Range (80-100%)
Focuses on extreme pruning scenarios:
```python
prune_range=(0.80, 1.0, 0.02)  # 80%, 82%, 84%, ..., 100%
```

### Cross-Validation

The evaluation script automatically performs k-fold cross-validation:
- Default: 5 folds
- Each fold trains a new model with different random seed
- Results are aggregated with mean ± std

### Generated Plots

All plots are automatically saved to `results/evaluation_TIMESTAMP/`:

1. **accuracy_comparison.png** - Accuracy vs. pruning fraction
2. **size_comparison.png** - Model size vs. pruning fraction
3. **mcc_comparison.png** - MCC vs. pruning fraction (with confidence bands)
4. **precision_comparison.png** - Precision vs. pruning fraction (with confidence bands)
5. **auc_comparison.png** - Bar chart comparing mean AUCs

### Statistical Analysis

The script performs paired t-tests to compare pruning methods:
```
Statistical Comparison (Paired t-tests vs magnitude)
============================================================
magnitude vs fim                      : t=  2.345, p=0.0234 *
magnitude vs mag_fim_one_shot         : t=  3.456, p=0.0012 **
============================================================
Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant
```

## Configuration

Edit `src/config.yaml` to customize evaluation:

```yaml
evaluation:
  cross_validation:
    num_folds: 5  # Number of CV folds
    
  pruning_sweep:
    full_range:
      start: 0.0
      end: 1.0
      step: 0.05
    high_range:
      start: 0.80
      end: 1.0
      step: 0.02
  
  plots:
    save_plots: true
    plot_formats: ["png", "pdf"]
    dpi: 300
```

## Using Individual Functions

You can also use the evaluation functions programmatically:

```python
from src.utils.evaluation import (
    evaluate_accuracy,
    get_mcc_and_precision,
    calculate_auc,
    plot_accuracy_comparison
)

# Evaluate a model
accuracy = evaluate_accuracy(model, test_loader, device)
mcc, precision = get_mcc_and_precision(model, test_loader, device)

# Calculate AUC
auc = calculate_auc(pruning_percentages, accuracies)

# Create plots
plot_accuracy_comparison(
    prune_pcts=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    results_dict={
        'Magnitude': [0.98, 0.95, 0.90, 0.80, 0.60, 0.10],
        'FIM': [0.98, 0.96, 0.92, 0.85, 0.70, 0.20]
    },
    save_path='my_plot.png'
)
```

## Example: Custom Evaluation

```python
import torch
from src.models.simple_cnn import SimpleCNN
from src.pruning.magnitude_pruning import MagnitudePruner
from src.utils.evaluation import evaluate_accuracy, plot_accuracy_comparison

# Load model and data
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))

# Evaluate at different pruning levels
results = []
thresholds = [0.0, 0.01, 0.05, 0.1, 0.2]

for threshold in thresholds:
    pruner = MagnitudePruner(threshold=threshold)
    pruned_model = pruner.apply_pruning(model)
    acc = evaluate_accuracy(pruned_model, test_loader)
    results.append(acc)

# Plot
plot_accuracy_comparison(
    prune_pcts=thresholds,
    results_dict={'My Method': results},
    title='Custom Evaluation'
)
```

## Output Structure

```
results/
└── evaluation_20241114_153045/
    ├── model_fold_0.pth
    ├── model_fold_1.pth
    ├── model_fold_2.pth
    ├── model_fold_3.pth
    ├── model_fold_4.pth
    ├── accuracy_comparison.png
    ├── size_comparison.png
    ├── mcc_comparison.png
    ├── precision_comparison.png
    ├── auc_comparison.png
    └── results_summary.txt
```

## Tips

1. **Memory Management**: The evaluation creates many model copies. If you run out of memory:
   - Reduce `num_folds`
   - Increase `step` in pruning ranges (fewer pruning levels)
   - Use smaller batch sizes

2. **Speed**: Full evaluation can take time. For quick tests:
   - Reduce `num_epochs` for training
   - Use fewer `num_folds`
   - Test on smaller pruning ranges

3. **Comparing Methods**: To add a new pruning method:
   - Implement it following the `BasePruner` interface
   - Add it to `all_results` dict in `evaluate_pruning.py`
   - Add corresponding evaluation sweep

## Requirements

Make sure you have all dependencies:
```bash
pip install torch torchvision matplotlib numpy scikit-learn scipy pyyaml
```