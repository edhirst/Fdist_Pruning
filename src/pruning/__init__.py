"""
The pruning schemes, one module each.

`pruning.pruning_scheme` in the config selects one by name; `build_pruner` in
src/run_pruning.py maps that name to the class:

    magnitude          MagnitudePruner        |w|
    fim                FIMPruner              I_kk
    f_dist_one_shot    FDistOneShotPruner     sqrt(I_kk(theta*)) |w|
    f_dist_iterative   FDistIterativePruner   sqrt(I_kk(theta))  |w|
    f_dist_global      FDistGlobalPruner      |w| mean_a sqrt(I_kk(a theta))
    f_dist             FDistPruner            sqrt(mean_a I_kk(theta|w -> a w)) |w|

The last four are the Fisher-distance family, in increasing order of how
faithfully they resolve the metric along the path from a weight to zero. All
subclass BasePruner and share the eligibility policy in prunable.py, so each is
free to differ only in how it scores a weight.
"""
# src/pruning/__init__.py

from .magnitude_pruning import MagnitudePruner
from .fim_pruning import FIMPruner
from .f_dist_one_shot import FDistOneShotPruner
from .f_dist_iterative import FDistIterativePruner
from .f_dist import FDistPruner
from .f_dist_global import FDistGlobalPruner
from .prunable import get_prunable_mask, resolve_prunable_exclude

__all__ = [
    "MagnitudePruner",
    "FIMPruner",
    "FDistOneShotPruner",
    "FDistIterativePruner",
    "FDistPruner",
    "FDistGlobalPruner",
    "get_prunable_mask",
    "resolve_prunable_exclude",
]
