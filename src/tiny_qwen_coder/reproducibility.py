"""Deterministic seeding helpers shared by data preparation, training, and evaluation."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_MAX_SEED = 2**32 - 1
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class SeedError(ValueError):
    """Raised when a seed or deterministic-runtime setting is invalid."""


def validate_seed(seed: int) -> None:
    """Validate the project-wide seed range shared by Python, NumPy, and PyTorch."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SeedError("seed must be an integer")
    if not 0 <= seed <= _MAX_SEED:
        raise SeedError(f"seed must be between 0 and {_MAX_SEED}")


@dataclass(frozen=True, slots=True)
class SeedSettings:
    """Determinism settings established by :func:`seed_everything`."""

    seed: int
    python_hash_seed: str
    cublas_workspace_config: str
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_seeded: bool


def seed_everything(seed: int) -> SeedSettings:
    """Seed Python, NumPy, and PyTorch and enable fail-closed deterministic algorithms.

    This function should run before dataset shuffling and before CUDA work begins. Setting
    ``PYTHONHASHSEED`` here makes child processes deterministic; the current interpreter's
    hash randomization is fixed at interpreter startup and cannot be changed retroactively.
    """

    validate_seed(seed)

    # Import the numeric runtimes lazily so lightweight config validation can reuse
    # validate_seed() without importing PyTorch or initializing CUDA-related modules.
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return SeedSettings(
        seed=seed,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_seeded=cuda_available,
    )


def make_torch_generator(seed: int) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for deterministic DataLoader/sampler use."""

    validate_seed(seed)
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_torch_worker(_worker_id: int) -> None:
    """Seed Python and NumPy inside a PyTorch DataLoader worker.

    PyTorch assigns each worker a deterministic initial seed when the DataLoader receives
    the generator returned by :func:`make_torch_generator`.
    """

    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
