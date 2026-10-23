"""Expand qd model fields to per-world arrays for domain randomization.

The former Warp version tiled with a dedicated wp kernel; field expansion
runs once at model-build time, so a numpy tile plus ``mujoco_qd``'s
``from_data`` allocation is semantically identical and simpler.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd
from mujoco_qd._src.warp_util import from_data


def _element_dtype(tail: tuple[int, ...]) -> Any:
  """The exact compound dtype mujoco_qd kernels annotate for ``tail``.

  VectorNdarray reports a scalar ``.dtype``, so the compound type is recovered
  from ``mujoco_qd._src.types`` (the same type objects the engine kernels
  annotate — reusing them keeps the kernel ABI identical; structurally-built
  types pass the mapper but are not ABI-safe).
  """
  import quadrants as qd
  from mujoco_qd import _src as _mq

  # 2/3/4-extent vectors live in qd.math; the engine's wider compound types
  # live in mujoco_qd._src.types.
  modules = (qd.math, _mq.types)
  if len(tail) == 1:
    name = f"vec{tail[0]}"
  else:
    name = f"mat{tail[0]}{tail[1]}"
  for mod in modules:
    cand = getattr(mod, name, None)
    if cand is not None and type(cand).__name__ in ("VectorType", "MatrixType"):
      return cand
  types_mod = _mq.types
  # Fallback: unique structural match (types like spatial_vector alias the
  # same extents as vec3, so only accept when unambiguous).
  matches = []
  for name in dir(types_mod):
    cand = getattr(types_mod, name)
    kind = type(cand).__name__
    if kind == "VectorType" and len(tail) == 1 and cand.n == tail[0]:
      matches.append(cand)
    if kind == "MatrixType" and len(tail) == 2 and (cand.n, cand.m) == tail:
      matches.append(cand)
  unique = {id(m): m for m in matches}
  if len(unique) == 1:
    return next(iter(unique.values()))
  raise TypeError(f"No unambiguous mujoco_qd compound dtype for element tail {tail}")


def expand_model_fields(model: Any, nworld: int, fields_to_expand: list[str]) -> None:
  if nworld == 1:
    return

  def tile(x: Any) -> Any:
    # Element-major numpy layout (vec/mat fields carry trailing extents that
    # are NOT part of the qd array's shape).
    src = x.to_numpy()
    reps = (nworld,) + (1,) * (src.ndim - 1)
    data = np.tile(src, reps)
    tail = src.shape[len(x.shape):]
    dtype = _element_dtype(tail) if tail else x.dtype
    return from_data(data, dtype=dtype)

  for field in model.__dataclass_fields__:
    if field in fields_to_expand:
      array = getattr(model, field)
      if array.shape[0] != 1:
        continue  # Already per-world (e.g., variant compilation).
      setattr(model, field, tile(array))
