# Fdist_NNPruning
Code for implementation of the novel Fisher-distance pruning scheme, with its application to Neural Networks.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
   ...the `cd` into it locally.

2. Follow the instructions in the `environment` folder to set up the venv.

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