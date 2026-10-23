import os
import random

import numpy as np
import torch


def seed_rng(seed: int, torch_deterministic: bool = False) -> None:
  """Seed all random number generators for reproducibility.

  The qd engine's physics is deterministic given identical inputs, so no
  engine-side RNG seeding is needed.
  """
  os.environ["PYTHONHASHSEED"] = str(seed)

  random.seed(seed)
  np.random.seed(seed)

  # Ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
  torch.manual_seed(seed)  # Seed RNG for all devices.
  # Use deterministic algorithms when possible.
  if torch_deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
