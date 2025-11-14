# MNIST and Fashion MNIST Pruning Framework

This project implements various pruning schemes for training models on the MNIST and Fashion MNIST datasets. The goal is to minimize training time while maintaining model performance through different pruning techniques.

## Project Structure

```
mnist-fmnist-pruning
├── src
│   ├── main.py               # Entry point for the application
│   ├── train.py              # Training loop and logic
│   ├── config.yaml           # Main configuration file
│   ├── pruning               # Directory containing pruning schemes
│   │   ├── __init__.py
│   │   ├── base_pruner.py    # Abstract base class for pruning
│   │   ├── magnitude_pruning.py
│   │   ├── fim_pruning.py
│   │   ├── magnitude_fim_one_shot.py
│   │   ├── magnitude_fim_iterative.py
│   │   └── sqrt_averaged_magnitude_fim.py
│   ├── models                # Directory containing model architectures
│   │   ├── __init__.py
│   │   └── simple_cnn.py
│   └── utils                 # Utility functions
│       ├── __init__.py
│       ├── data_loader.py
│       └── metrics.py
├── config                    # Configuration files
│   ├── datasets.yaml
│   ├── models.yaml
│   └── pruning.yaml
├── notebooks                 # Jupyter notebooks for experimentation
│   └── experiments.ipynb
├── requirements.txt          # Project dependencies
├── .gitignore                # Files to ignore in version control
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mnist-fmnist-pruning
   ```

2. Create a virtual environment and activate it:
   ```
   conda create -n pruning-env python=3.8
   conda activate pruning-env
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Configure the parameters in the `config/config.yaml`, `config/datasets.yaml`, `config/models.yaml`, and `config/pruning.yaml` files according to your needs.

2. Run the main script to start training:
   ```
   python src/main.py
   ```

## Pruning Schemes

The project implements the following pruning schemes:

- **Magnitude Pruning**: Removes weights based on their magnitude.
- **FIM Pruning**: Utilizes the Fisher Information Matrix for pruning.
- **Magnitude x FIM Pruning (One Shot)**: Combines magnitude and FIM pruning in a single pass.
- **Magnitude x FIM Pruning (Iterative)**: Applies magnitude and FIM pruning iteratively.
- **Square Root of Averaged Magnitude x FIM**: Uses the square root of the averaged values for pruning.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.