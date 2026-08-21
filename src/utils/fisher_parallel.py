"""
Process-sharded driver for the forward-mode Fisher probes.

Exact f_dist evaluates one independent probe per (coordinate, alpha). The work is
compute-bound per core -- measured on SimpleViT/CIFAR-10, a single probe costs
~1.35 s on one core and torch's intra-op threading saturates at about 4 threads
(1.40 s at 1 thread, 0.86 s at 4, 0.89 s at 8, and neither `probe_batch` nor
`sample_chunk` changes that). The only way to use a 128-core node is therefore to
run many single-threaded processes, one slice of the coordinate list each.

Sharding is by COORDINATE, never by sample, so each coordinate still accumulates
over the whole Fisher subset in the original batch order. Together with shard
boundaries snapped to probe_batch multiples (see _slices), that makes the result
bit-identical to the serial path at any worker count, for a fixed probe_batch --
asserted in tests/test_fisher_sharding_equivalence.py, not merely assumed.
(Changing probe_batch itself does perturb the last ulp, since it changes the vmap
batch shape and hence BLAS blocking; that is unrelated to sharding.)

CPU only: forking a CUDA context is unsafe, and this exists for the big CPU
queues anyway (see hpc/README.md).
"""
import os

import torch
import torch.multiprocessing as mp

from .fim_calculator import fisher_entries_forward

# Worker-side state, populated by _init in each forked child. Passing the model
# through the fork (copy-on-write) rather than pickling it per task keeps the
# per-task payload down to a slice of indices.
_W = {}


def resolve_workers(config_value=None):
    """
    Number of probe worker processes.

    FDIST_WORKERS (env) wins, else PBS_NP (what OpenPBS exports for the job's
    core count), else the config value, else 1. Default 1 means the serial path
    unless something explicitly asks for more, so nothing changes silently.
    """
    for env in ("FDIST_WORKERS", "PBS_NP"):
        v = os.environ.get(env)
        if v is not None and str(v).strip() != "":
            try:
                n = int(v)
            except ValueError:
                continue
            if n > 0:
                return n
    if config_value:
        try:
            return max(1, int(config_value))
        except (TypeError, ValueError):
            return 1
    return 1


def materialise_batches(loader):
    """
    Pull a DataLoader into a plain list of (x, y) CPU tensors.

    Two reasons. (1) Workers then never touch DataLoader machinery, which does
    not survive forking cleanly when the loader itself has workers. (2) The
    serial path re-iterates the loader once per (tensor, alpha) -- for the ViT
    that is ~100 passes per pruning step, each re-decoding and re-transforming
    every image. Materialising once removes that entirely.
    """
    return [(xb.detach().cpu(),
             yb.detach().cpu() if torch.is_tensor(yb) else yb)
            for xb, yb in loader]


def _init(model, batches, probe_batch, sample_chunk):
    # One thread per worker: N processes each spawning N threads would oversubscribe
    # the node and run slower than serial.
    torch.set_num_threads(1)
    _W["model"] = model
    _W["batches"] = batches
    _W["probe_batch"] = probe_batch
    _W["sample_chunk"] = sample_chunk


def _task(args):
    tensor_name, idxs, alphas = args
    return fisher_entries_forward(
        _W["model"], _W["batches"], tensor_name, idxs, alphas,
        device="cpu", probe_batch=_W["probe_batch"], sample_chunk=_W["sample_chunk"],
    )


def _slices(n, workers, probe_batch, per_worker=4):
    """
    Split n probes into contiguous chunks for load balance.

    Boundaries are snapped to multiples of `probe_batch`. This is what makes the
    sharded result bit-identical to the serial one: fisher_entries_forward chunks
    its probes as [k*pb, (k+1)*pb), and a vmap over a different batch size picks a
    different BLAS blocking, which reorders the floating-point sums. Aligning the
    shards to those same boundaries means every shard replays exactly the batch
    shapes the serial path would have used. Without this the results agree only to
    ~1 ulp -- harmless for the pruning decision, but needlessly non-reproducible.
    """
    if n <= 0:
        return []
    pb = max(1, int(probe_batch))
    nblocks = -(-n // pb)                       # ceil: number of probe_batch blocks
    ntasks = max(1, min(nblocks, workers * per_worker))
    edges = [min(n, round(i * nblocks / ntasks) * pb) for i in range(ntasks + 1)]
    edges[-1] = n
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


class ProbePool:
    """
    A reusable pool of probe workers.

    Built once per pruning step (the model is constant within a step, so it can
    be inherited through the fork) and reused across every (tensor, alpha) task
    in that step -- forking 128 children per tensor would otherwise dominate.

    Falls back to running tasks inline when workers <= 1, so the same call site
    covers both paths.
    """

    def __init__(self, model, batches, workers=1, probe_batch=32, sample_chunk=None):
        self.workers = max(1, int(workers))
        self.probe_batch = probe_batch
        self.sample_chunk = sample_chunk
        self._pool = None
        if self.workers > 1:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(
                self.workers, initializer=_init,
                initargs=(model, batches, probe_batch, sample_chunk),
            )
        else:
            _init(model, batches, probe_batch, sample_chunk)

    def run(self, tensor_name, idxs, alphas):
        """
        Fisher entries for these coordinates of `tensor_name`, in the given order.

        Reassembled by slice index, so the returned tensor does not depend on the
        worker count or on completion order.
        """
        n = idxs.numel()
        if n == 0:
            return torch.empty(0)
        if self._pool is None:
            return _task((tensor_name, idxs, alphas))

        tasks = [(tensor_name, idxs[a:b], alphas[a:b])
                 for a, b in _slices(n, self.workers, self.probe_batch)]
        parts = self._pool.map(_task, tasks)
        return torch.cat(parts, dim=0)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
