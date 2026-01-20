# src/pruning/__init__.py

from .magnitude_pruning import MagnitudePruner
from .fim_pruning import FIMPruner
from .f_dist_one_shot import FDistOneShotPruner
from .f_dist_iterative import FDistIterativePruner
from .f_dist import FDistPruner

__all__ = [
    "MagnitudePruner",
    "FIMPruner",
    "FDistOneShotPruner",
    "FDistIterativePruner",
    "FDistPruner"
]