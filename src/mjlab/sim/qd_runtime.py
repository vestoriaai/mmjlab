"""Process-wide quadrants initialization authority.

quadrants is initialized once per process with exactly one arch; a second
``qd.init`` silently re-inits and invalidates every existing qd allocation.
Both the qd physics backend (arch implied by the torch device) and the qd
render/raycast backends (arch preference list) must go through
:func:`ensure_initialized`.
"""

from __future__ import annotations

_qd_arch: str | None = None


def current_arch() -> str | None:
  """The initialized qd arch name (cpu/metal/cuda/vulkan), or None."""
  return _qd_arch


def ensure_initialized(candidates: list[str]) -> str:
  """Initialize quadrants on the first available arch, unless already done.

  Args:
    candidates: qd arch names in preference order (e.g. ["cpu"],
      ["metal", "cuda", "vulkan", "cpu"]).

  Returns:
    The initialized arch name.

  Raises:
    RuntimeError: If quadrants is already initialized on an arch outside
      ``candidates``.
    ValueError: If no candidate arch can be initialized.
  """
  global _qd_arch
  import quadrants as qd

  if _qd_arch is not None:
    if _qd_arch in candidates:
      return _qd_arch
    raise RuntimeError(
      f"quadrants is already initialized for arch '{_qd_arch}', which is not "
      f"among the requested candidates {candidates}. One arch per process."
    )
  errors: list[str] = []
  for arch in candidates:
    try:
      qd.init(arch=getattr(qd, arch), cfg_optimization=False)
      _qd_arch = arch
      return arch
    except Exception as e:  # noqa: BLE001 - probe every candidate
      errors.append(f"{arch}: {e}")
  raise ValueError(f"Could not initialize quadrants on any of {candidates}: {errors}")
