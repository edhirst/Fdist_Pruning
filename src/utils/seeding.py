"""
Global seed control.

The paper table reports each entry as a mean +/- std over independent seeds, so
a seed has to reach everything that varies between repeats: parameter
initialisation, the training data order, dropout, and the Fisher subset draw.
`experiment.seed` in the config is the single knob; `resolve_seed` also honours
the FDIST_SEED environment variable so one config can be reused across a set of
HPC jobs that differ only by seed.
"""
import os
import random

import numpy as np
import torch


def resolve_seed(config):
    """
    Seed for this run, or None when none is configured.

    FDIST_SEED (env) wins over experiment.seed (config). Returning None when
    neither is set keeps every pre-existing single-run behaviour untouched:
    no reseeding, no per-seed checkpoint names, no change to the Fisher subset
    draw. Seeding is opt-in, so old results stay reproducible.
    """
    env = os.environ.get("FDIST_SEED")
    if env is not None and env.strip() != "":
        return int(env)
    exp = (config or {}).get("experiment", {}) or {}
    seed = exp.get("seed", None)
    return None if seed is None else int(seed)


def seed_everything(seed, deterministic=False):
    """
    Seed python / numpy / torch (all devices).

    deterministic=True additionally forces deterministic cuDNN kernels. It is off
    by default because it slows training noticeably and the run-to-run variation
    it removes is exactly what the +/- std column is measuring.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

