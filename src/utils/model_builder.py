"""
Shared model construction from config, used by train_model.py, run_pruning.py and
subset_experiment.py (previously three diverging copies).

model.common.model_type accepts 'nn' | 'cnn' | 'transformer' (case-insensitive;
legacy 'NN'/'CNN' spellings keep working).
"""
import os

from ..models.simple_nn import SimpleNN
from ..models.simple_cnn import SimpleCNN
from ..models.simple_vit import SimpleViT
from .data_loader import normalize_dataset_name


# img_size, in_channels per dataset (keys are canonical names from normalize_dataset_name)
DATASET_SHAPES = {
    "mnist": (28, 1),
    "fashion_mnist": (28, 1),
    "cifar10": (32, 3),
}


def normalize_model_type(model_type):
    mt = str(model_type).strip().lower()
    if mt == "nn":
        return "NN"
    if mt == "cnn":
        return "CNN"
    if mt in ("transformer", "vit"):
        return "Transformer"
    raise ValueError(f"Unknown model_type: {model_type!r}. Supported: 'nn', 'cnn', 'transformer'.")


def get_model_type(config):
    common_cfg = (config.get("model", {}) or {}).get("common", {}) or {}
    return normalize_model_type(common_cfg.get("model_type", "NN"))


def build_model_from_config(config, dataset_name=None):
    """
    Build the model described by config for the given dataset.

    Returns:
        (model, arch_name)
    """
    model_cfg = config.get("model", {}) or {}
    common_cfg = model_cfg.get("common", {}) or {}
    model_type = normalize_model_type(common_cfg.get("model_type", "NN"))
    num_classes = int(common_cfg.get("num_classes", 10))

    if dataset_name is None:
        dataset_name = (config.get("dataset", {}) or {}).get("name", "mnist")
    dataset_name = normalize_dataset_name(dataset_name)
    img_size, in_channels = DATASET_SHAPES[dataset_name]

    if model_type == "NN":
        nn_cfg = model_cfg.get("NN", {}) or {}
        hidden_size = int(nn_cfg.get("hidden_size", 32))
        hidden_layers = int(nn_cfg.get("hidden_layers", 2))
        input_size = in_channels * img_size * img_size
        model = SimpleNN(
            input_size=input_size, hidden_size=hidden_size,
            hidden_layers=hidden_layers, num_classes=num_classes,
        )
        arch_name = f"SimpleNN_h{hidden_layers}_n{hidden_size}"

    elif model_type == "CNN":
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels, img_size=img_size)
        arch_name = "SimpleCNN"

    else:  # Transformer
        t_cfg = model_cfg.get("Transformer", {}) or model_cfg.get("transformer", {}) or {}
        embed_dim = int(t_cfg.get("embed_dim", 128))
        depth = int(t_cfg.get("depth", 4))
        num_heads = int(t_cfg.get("num_heads", 4))
        mlp_ratio = float(t_cfg.get("mlp_ratio", 2.0))
        patch_size = int(t_cfg.get("patch_size", 4))
        dropout = float(t_cfg.get("dropout", 0.0))
        model = SimpleViT(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            dropout=dropout,
        )
        arch_name = f"SimpleViT_d{depth}_e{embed_dim}_h{num_heads}_p{patch_size}"

    return model, arch_name


def resolve_checkpoint_path(config, arch_name, dataset_name, seed=None):
    """
    Explicit paths.pretrained_model_path wins (backward compatible).
    When it is null/empty, default to {model_save_path}/{arch_name}_{dataset}.pth,
    or {arch_name}_{dataset}_seed{N}.pth for a multi-seed run.

    `seed` must be None for single-run behaviour: multi-seed jobs each train and
    prune their own checkpoint, so the filenames must not collide.
    """
    paths_cfg = config.get("paths", {}) or {}
    explicit = paths_cfg.get("pretrained_model_path", None)
    if explicit:
        return explicit
    save_dir = paths_cfg.get("model_save_path", "models/")
    tag = "" if seed is None else f"_seed{int(seed)}"
    return os.path.join(save_dir, f"{arch_name}_{normalize_dataset_name(dataset_name)}{tag}.pth")
