import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train_model

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config('src/config.yaml')
    
    # Initialize training with the loaded configuration
    train_model(config)

if __name__ == "__main__":
    main()