import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from .utils.data_loader import load_mnist, load_fashion_mnist
from .models.simple_cnn import SimpleCNN
from .models.simple_nn import SimpleNN 


# load config from yaml
def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

# Use GPU if available else CPU
def get_device():
    # CUDA first, then MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



# ------------------------------- Training Script -------------------------------

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
        Train model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy



def build_dataloaders(config):
    """
        Load dataset based on config
    """
    dataset_name = config.get("dataset", {}).get("name", "mnist").lower()
    batch_size = config["training"]["batch_size"]

    if dataset_name == "mnist":
        train_loader = load_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_mnist(batch_size=batch_size, train=False, download=True)
    elif dataset_name == "fashion_mnist":
        train_loader = load_fashion_mnist(batch_size=batch_size, train=True, download=True)
        test_loader = load_fashion_mnist(batch_size=batch_size, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_name, train_loader, test_loader



def build_model(config):
    model_cfg = config.get("model", {})
    common_cfg = model_cfg.get("common", {})

    model_type = common_cfg.get("model_type", "NN")
    num_classes = common_cfg.get("num_classes", 10)

    if model_type == "NN":
        nn_cfg = model_cfg.get("NN", {})
        hidden_size = nn_cfg.get("hidden_size", 32)
        hidden_layers = nn_cfg.get("hidden_layers", 2)
        model = SimpleNN(hidden_size=hidden_size, hidden_layers=hidden_layers, num_classes=num_classes)
        arch_name = f"SimpleNN_h{hidden_layers}_n{hidden_size}"

    elif model_type == "CNN":
        cnn_cfg = model_cfg.get("CNN", {}) or {}
        # If your SimpleCNN accepts extra args, read from cnn_cfg here.
        # For now, only num_classes:
        model = SimpleCNN(num_classes=num_classes)
        arch_name = f"SimpleCNN"

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: 'NN', 'CNN'.")

    return model, arch_name


def build_optimizer(config, model):
    learning_rate = config["training"]["learning_rate"]
    optimizer_name = config["training"].get("optimizer", "adam").lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        # default
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer


def make_model_filename(config, arch_name: str):
    tr_cfg = config.get("training", {})
    # Example:
    # SimpleNN_h2_n32.pth
    return f"{arch_name}.pth"




def train_model(config, device):
    """
        Main training function that takes config as input
    """

    dataset_name, train_loader, test_loader = build_dataloaders(config)
    model, arch_name = build_model(config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]

    # Train the model

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    print("-" * 50)
    print("Training completed.")

    # ---- Test evaluation ----
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    
    # Save model if path specified
    model_save_path = config.get('paths', {}).get('model_save_path')
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        filename = make_model_filename(config, arch_name)
        save_file = os.path.join(model_save_path, filename)
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to: {save_file}")
    
    return model


if __name__ == "__main__":

    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/config.yaml"
    config = load_config(config_path)

    device = get_device()
    train_model(config, device=device)