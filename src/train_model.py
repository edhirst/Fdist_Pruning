import os
import sys
import math
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from .utils.data_loader import build_dataloaders
from .utils.model_builder import build_model_from_config, get_model_type, resolve_checkpoint_path


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

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, grad_clip=None):
    """
        Train model for one epoch.
        scheduler (if given) is stepped per batch; grad_clip (if given) applies
        gradient-norm clipping — both are used by the transformer recipe only.
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
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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


def get_training_cfg(config, model_type):
    """
    Flat training keys are the base defaults; an optional per-architecture block
    (training.transformer / training.nn / training.cnn) overrides them for that
    model type.
    """
    override_key = {"Transformer": "transformer", "NN": "nn", "CNN": "cnn"}[model_type]
    tr_cfg = dict(config.get("training", {}) or {})
    overrides = tr_cfg.get(override_key, {}) or {}
    for key in ("transformer", "nn", "cnn"):
        tr_cfg.pop(key, None)
    tr_cfg.update(overrides)
    return tr_cfg


def build_optimizer(tr_cfg, model):
    learning_rate = tr_cfg["learning_rate"]
    optimizer_name = str(tr_cfg.get("optimizer", "adam")).lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        weight_decay = float(tr_cfg.get("weight_decay", 0.05))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        # default
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, num_epochs, steps_per_epoch):
    """Linear warmup then cosine decay, stepped per batch (standard ViT recipe)."""
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = max(1, int(num_epochs * steps_per_epoch))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(config, device):
    """
        Main training function that takes config as input
    """
    model_type = get_model_type(config)

    # augmentation is train-split-only and currently applies to CIFAR-10
    dataset_name, train_loader, test_loader = build_dataloaders(config, augment=True)
    model, arch_name = build_model_from_config(config, dataset_name=dataset_name)
    model = model.to(device)

    tr_cfg = get_training_cfg(config, model_type)

    label_smoothing = float(tr_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(tr_cfg, model)

    num_epochs = int(tr_cfg["num_epochs"])
    batch_size = tr_cfg["batch_size"]
    learning_rate = tr_cfg["learning_rate"]
    grad_clip = tr_cfg.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip else None

    scheduler = None
    warmup_epochs = tr_cfg.get("warmup_epochs", None)
    if warmup_epochs is not None:
        scheduler = build_warmup_cosine_scheduler(
            optimizer, float(warmup_epochs), num_epochs, len(train_loader)
        )

    # Train the model

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {arch_name} ({n_params:,} params)")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {tr_cfg.get('optimizer', 'adam')}"
          + (f" | warmup {warmup_epochs} ep + cosine" if scheduler is not None else "")
          + (f" | label_smoothing {label_smoothing}" if label_smoothing else "")
          + (f" | grad_clip {grad_clip}" if grad_clip else ""))
    print("-" * 50)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, grad_clip=grad_clip,
        )
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    print("-" * 50)
    print("Training completed.")

    # ---- Test evaluation ----
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


    # Save model where run_pruning.py will look for it
    save_file = resolve_checkpoint_path(config, arch_name, dataset_name)
    os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to: {save_file}")

    return model


if __name__ == "__main__":

    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/config.yaml"
    config = load_config(config_path)

    device = get_device()
    train_model(config, device=device)
