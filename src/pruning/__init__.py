# src/pruning/__init__.py

from .magnitude_pruning import MagnitudePruner
from .fim_value import fim_value
from .fim_pruning import FIMPruner

__all__ = [
    "Magnitude",
    "fim_value",
    "FIMPruner"
]