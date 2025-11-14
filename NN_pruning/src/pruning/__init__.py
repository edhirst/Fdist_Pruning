# src/pruning/__init__.py

from .magnitude_pruning import MagnitudePruner
from .fim_pruning import FIMPruner
from .magnitude_fim_one_shot import MagnitudeFIMOneShotPruner
from .magnitude_fim_iterative import MagnitudeFIMIterativePruner
from .sqrt_averaged_magnitude_fim import SqrtAveragedMagnitudeFIMPruner

__all__ = [
    "MagnitudePruner",
    "FIMPruner",
    "MagnitudeFIMOneShotPruner",
    "MagnitudeFIMIterativePruner",
    "SqrtAveragedMagnitudeFIMPruner",
]