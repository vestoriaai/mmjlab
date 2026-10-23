from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import mujoco
import mujoco_qd as mjwarp  # mjwarp-shaped API: the qd port of MuJoCo Warp.
import numpy as np
import torch
from mujoco_qd._src.warp_util import from_data

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.qd_runtime import ensure_initialized
from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import QdBridge, TorchArray
from mjlab.utils.nan_guard import NanGuard, NanGuardCfg

if TYPE_CHECKING:
  from mjlab.entity.variants import VariantMetadata
  from mjlab.sensor.sensor_context_cpu import SensorContextCPU

# Type aliases for better IDE support while maintaining runtime compatibility
# At runtime, QdBridge wraps the actual mujoco_qd objects.
if TYPE_CHECKING:
  ModelBridge = mjwarp.Model
  DataBridge = mjwarp.Data
else:
  ModelBridge = QdBridge
  DataBridge = QdBridge

# qd arch candidates for each torch device string. mjlab talks to the engine
# through torch-side bridges, so the torch device selects the qd arch.
_DEVICE_TO_ARCH_CANDIDATES: dict[str, list[str]] = {
  "cpu": ["cpu"],
  "mps": ["metal"],
  "metal": ["metal"],
}


def _ensure_qd_init(device: str) -> None:
  """Initialize quadrants once, on the arch implied by ``device``."""
  candidates = _DEVICE_TO_ARCH_CANDIDATES.get(device)
  if candidates is None:
    raise ValueError(
      f"Device '{device}' is not supported by the qd backend "
      f"(available: {sorted(_DEVICE_TO_ARCH_CANDIDATES)})."
    )
  ensure_initialized(candidates)


_JACOBIAN_MAP = {
  "auto": mujoco.mjtJacobian.mjJAC_AUTO,
  "dense": mujoco.mjtJacobian.mjJAC_DENSE,
  "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}
_CONE_MAP = {
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
_INTEGRATOR_MAP = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
  "rk4": mujoco.mjtIntegrator.mjINT_RK4,
  "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
}
# Integrators only available on some MuJoCo versions.
for _name in ("discrete",):
  _enum = getattr(mujoco.mjtIntegrator, f"mjINT_{_name.upper()}", None)
  if _enum is not None:
    _INTEGRATOR_MAP[_name] = _enum
_SOLVER_MAP = {
  "newton": mujoco.mjtSolver.mjSOL_NEWTON,
  "cg": mujoco.mjtSolver.mjSOL_CG,
  "pgs": mujoco.mjtSolver.mjSOL_PGS,
}
_BROADPHASE_MAP = {
  "nxn": mjwarp.BroadphaseType.NXN,
  "sap_tile": mjwarp.BroadphaseType.SAP_TILE,
  "sap_segmented": mjwarp.BroadphaseType.SAP_SEGMENTED,
}
_BROADPHASE_FILTER_MAP = {
  "plane": mjwarp.BroadphaseFilter.PLANE,
  "sphere": mjwarp.BroadphaseFilter.SPHERE,
  "aabb": mjwarp.BroadphaseFilter.AABB,
  "obb": mjwarp.BroadphaseFilter.OBB,
}

# Maps short flag names to MuJoCo enum values.
# Names match the XML <flag> attribute names (e.g. <flag contact="disable"/>).
_DISABLE_FLAG_MAP: dict[str, int] = {
  name.removeprefix("mjDSBL_").lower(): getattr(mujoco.mjtDisableBit, name).value
  for name in dir(mujoco.mjtDisableBit)
  if name.startswith("mjDSBL_")
}
_ENABLE_FLAG_MAP: dict[str, int] = {
  name.removeprefix("mjENBL_").lower(): getattr(mujoco.mjtEnableBit, name).value
  for name in dir(mujoco.mjtEnableBit)
  if name.startswith("mjENBL_")
}


@dataclass
class MujocoCfg:
  """Configuration for MuJoCo simulation parameters."""

  # Integrator settings.
  timestep: float = 0.002
  integrator: Literal["euler", "implicitfast", "rk4", "implicit", "discrete"] = (
    "implicitfast"
  )
  """Numerical integrator. The qd backend implements every integrator MuJoCo
  exposes (``discrete`` additionally requires a MuJoCo version that provides
  it)."""

  # Friction settings.
  impratio: float = 1.0
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"

  # Solver settings.
  jacobian: Literal["auto", "dense", "sparse"] = "auto"
  solver: Literal["newton", "cg", "pgs"] = "newton"
  iterations: int = 100
  tolerance: float = 1e-8
  ls_iterations: int = 50
  ls_tolerance: float = 0.01
  ccd_iterations: int = 50

  # Other.
  gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
  # Global MuJoCo option flags. Names match the XML <flag> attributes
  # (e.g. "contact", "gravity", "sensor"). See mjtDisableBit / mjtEnableBit.
  disableflags: tuple[str, ...] = ()
  """Disable flags to set (e.g. ``("contact",)`` to disable contacts)."""
  enableflags: tuple[str, ...] = ()
  """Enable flags to set (e.g. ``("energy",)`` to enable energy computation)."""

  def apply(self, model: mujoco.MjModel) -> None:
    """Apply configuration settings to a compiled MjModel."""
    if self.integrator not in _INTEGRATOR_MAP:
      raise ValueError(
        f"Integrator '{self.integrator}' is not available in MuJoCo "
        f"{mujoco.__version__}. Available: {sorted(_INTEGRATOR_MAP)}"
      )
    model.opt.jacobian = _JACOBIAN_MAP[self.jacobian]
    model.opt.cone = _CONE_MAP[self.cone]
    model.opt.integrator = _INTEGRATOR_MAP[self.integrator]
    model.opt.solver = _SOLVER_MAP[self.solver]
    model.opt.timestep = self.timestep
    model.opt.impratio = self.impratio
    model.opt.gravity[:] = self.gravity
    model.opt.iterations = self.iterations
    model.opt.tolerance = self.tolerance
    model.opt.ls_iterations = self.ls_iterations
    model.opt.ls_tolerance = self.ls_tolerance
    model.opt.ccd_iterations = self.ccd_iterations
    for flag in self.disableflags:
      if flag not in _DISABLE_FLAG_MAP:
        raise ValueError(
          f"Unknown disable flag {flag!r}. Valid flags: {sorted(_DISABLE_FLAG_MAP)}"
        )
      model.opt.disableflags |= _DISABLE_FLAG_MAP[flag]
    for flag in self.enableflags:
      if flag not in _ENABLE_FLAG_MAP:
        raise ValueError(
          f"Unknown enable flag {flag!r}. Valid flags: {sorted(_ENABLE_FLAG_MAP)}"
        )
      model.opt.enableflags |= _ENABLE_FLAG_MAP[flag]


@dataclass(kw_only=True)
class SimulationCfg:
  backend: Literal["qd", "warp", "classic"] = "qd"
  """Physics backend.

  ``"qd"`` (default) batches all environments in quadrants kernels (the
  ``mujoco_qd`` port of MuJoCo Warp) on the device implied by ``device``
  (``"cpu"`` or ``"mps"``). ``"warp"`` is accepted as an alias of ``"qd"``.
  ``"classic"`` steps per-environment ``mujoco.MjData`` with the classic C
  engine on CPU threads; it requires ``device="cpu"``. See
  :class:`mjlab.sim.classic.ClassicSimulation` for its limitations
  (no mesh variants)."""
  num_threads: int | None = None
  """Classic backend only: threads in mjbatch's pool.

  ``None`` (or ``0``) uses every logical CPU, mjbatch's default; a second thread per
  core measures faster for MuJoCo's latency-bound linear algebra."""
  nconmax: int | None = None
  """Number of contacts to allocate per world.

  Contacts exist in large heterogenous arrays: one world may have more than nconmax
  contacts. If None, a heuristic value is used."""
  njmax: int | None = None
  """Number of constraints to allocate per world.

  Constraint arrays are batched by world: no world may have more than njmax
  constraints. If None, a heuristic value is used."""
  include_efc: bool = False
  """Classic backend only: capture the constraint solver rows (``data.efc``).

  Off by default — capture costs a small per-step copy. When on,
  ``sim.data.efc`` exposes ``nefc`` (N,), ``type`` / ``id`` (N, njmax) int32 and
  ``force`` (N, njmax) float32, with the first ``nefc`` rows valid per world, so
  friction-dependent actuators can read the solver's per-row output. Raises at
  runtime if a world needs more than ``njmax`` rows."""
  contact_sensor_maxmatch: int = 64
  broadphase: Literal["nxn", "sap_tile", "sap_segmented"] | None = None
  """Broadphase collision algorithm. If None, use the engine default."""
  broadphase_filter: tuple[Literal["plane", "sphere", "aabb", "obb"], ...] | None = None
  """Bounding-volume filters applied during broadphase collision checking.

  If None, use the engine default."""
  ls_parallel: bool | None = None
  """Deprecated and ignored. Parallel linesearch was removed in MuJoCo Warp 3.10."""
  mujoco: MujocoCfg = field(default_factory=MujocoCfg)
  nan_guard: NanGuardCfg = field(default_factory=NanGuardCfg)

  def __post_init__(self) -> None:
    if self.ls_parallel is not None:
      warnings.warn(
        "SimulationCfg.ls_parallel is deprecated and ignored; parallel "
        "linesearch was removed in MuJoCo Warp 3.10.",
        DeprecationWarning,
        stacklevel=2,
      )
    if self.backend == "warp":
      # Backwards-compatible alias: "warp" now selects the qd engine.
      self.backend = "qd"

  def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
    """Apply engine-only settings to the qd model's Option (post ``put_model``)."""
    wp_opt.contact_sensor_maxmatch = self.contact_sensor_maxmatch
    if self.broadphase is not None:
      wp_opt.broadphase = _BROADPHASE_MAP[self.broadphase]
    if self.broadphase_filter is not None:
      wp_opt.broadphase_filter = mjwarp.BroadphaseFilter(0)
      for name in self.broadphase_filter:
        wp_opt.broadphase_filter |= _BROADPHASE_FILTER_MAP[name]


class Simulation:
  """Batched MuJoCo simulation powered by mujoco_qd (quadrants kernels).

  qd has a single implicit device per process (selected by ``device``:
  ``"cpu"`` -> ``qd.cpu``, ``"mps"``/``"metal"`` -> ``qd.metal``), so there
  are no explicit device scopes. qd does not support graph capture; ``step``/
  ``forward``/``reset`` launch their kernels directly.
  """

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    model: mujoco.MjModel | None = None,
    device: str = "cpu",
    *,
    spec: mujoco.MjSpec | None = None,
    variant_info: list[tuple[str, "VariantMetadata"]] | None = None,
  ):
    if variant_info:
      raise NotImplementedError(
        "The qd backend does not support mesh variants (per-world mesh "
        "assignment); variant realization is unsupported on every mmjlab "
        "backend."
      )
    _ensure_qd_init(device)
    self.cfg = cfg
    self.device = device
    self.num_envs = num_envs
    self._default_model_fields: dict[str, torch.Tensor] = {}
    # Fields whose DR baseline is per-world (DR's `_select_default_values`
    # uses this to know whether to index `[env, ...]` vs `[...]`).
    self._per_world_default_fields: set[str] = set()
    # Fields that have per-world arrays (viewer sync uses this to know what to
    # copy per env).
    self._expanded_fields: set[str] = set()
    # Per-entity variant assignment, keyed by entity name (no trailing
    # slash). Empty for non-variant scenes.
    self._world_to_variant: dict[str, torch.Tensor] = {}

    if spec is not None:
      compiled = spec.compile()
      cfg.mujoco.apply(compiled)
      self._init_with_model(compiled)
    elif model is not None:
      cfg.mujoco.apply(model)
      self._init_with_model(model)
    else:
      raise ValueError("Either model or spec must be provided.")

  def _init_with_model(self, model: mujoco.MjModel) -> None:
    """Standard path: build qd model from compiled MjModel."""
    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    self._wp_model = mjwarp.put_model(self._mj_model)
    self.cfg.apply_wp_opt(self._wp_model.opt)
    self._finish_init()

  def _finish_init(self) -> None:
    """Common initialization after the qd model is created."""
    self._wp_data = mjwarp.put_data(
      self._mj_model,
      self._mj_data,
      nworld=self.num_envs,
      nconmax=self.cfg.nconmax,
      njmax=self.cfg.njmax,
    )

    self._reset_mask_qd = from_data(
      np.zeros(self.num_envs, dtype=bool), dtype=bool
    )
    self._reset_mask = TorchArray(self._reset_mask_qd)

    self._model_bridge = QdBridge(self._wp_model, nworld=self.num_envs)
    self._data_bridge = QdBridge(self._wp_data)
    self._sensor_context: SensorContextCPU | None = None

    self.use_cuda_graph = False

    self.nan_guard = NanGuard(self.cfg.nan_guard, self.num_envs, self._mj_model)

  def create_graph(self) -> None:
    """No-op: qd does not support graph capture.

    Retained for interface parity with the classic backend; call sites
    (``expand_model_fields``) keep working unchanged.
    """

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def wp_model(self) -> mjwarp.Model:
    """Engine-native model (``mujoco_qd``; named ``wp_model`` for parity)."""
    return self._wp_model

  @property
  def wp_data(self) -> mjwarp.Data:
    """Engine-native data (``mujoco_qd``; named ``wp_data`` for parity)."""
    return self._wp_data

  @property
  def data(self) -> DataBridge:
    return cast("DataBridge", self._data_bridge)

  @property
  def model(self) -> ModelBridge:
    return cast("ModelBridge", self._model_bridge)

  @property
  def default_model_fields(self) -> dict[str, torch.Tensor]:
    """Default values for expanded model fields, used in domain randomization."""
    return self._default_model_fields

  @property
  def expanded_fields(self) -> set[str]:
    """Names of model fields that have been expanded for per-env DR."""
    return self._expanded_fields

  @property
  def per_world_default_fields(self) -> set[str]:
    """Fields with per-world defaults from variant compilation."""
    return self._per_world_default_fields

  @property
  def world_to_variant(self) -> dict[str, torch.Tensor]:
    """Per-entity variant assignment for variant scenes (empty: unsupported)."""
    return self._world_to_variant

  # Methods.

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """Expand model fields to support per-environment parameters."""
    if not fields:
      return

    invalid_fields = [f for f in fields if not hasattr(self._mj_model, f)]
    if invalid_fields:
      raise ValueError(f"Fields not found in model: {invalid_fields}")

    expand_model_fields(self._wp_model, self.num_envs, list(fields))
    self._expanded_fields.update(fields)
    self._model_bridge.clear_cache()

  def get_default_field(self, field: str) -> torch.Tensor:
    """Get the default value for a model field, caching for reuse.

    Returns the original values from the C MuJoCo model (mj_model), obtained
    from the final compiled scene spec before any randomization is applied.
    Not to be confused with the qd model (wp_model) which may have
    randomized values.
    """
    if field not in self._default_model_fields:
      if not hasattr(self._mj_model, field):
        raise ValueError(f"Field '{field}' not found in model")
      model_field = getattr(self.model, field)
      default_value = getattr(self._mj_model, field)
      self._default_model_fields[field] = torch.as_tensor(
        default_value, dtype=model_field.dtype, device=self.device
      ).clone()
    return self._default_model_fields[field]

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """Recompute derived model constants after domain randomization.

    Args:
      level: Which constants to recompute. ``set_const`` is the most
        expensive (covers body_mass changes), ``set_const_0`` covers
        qpos0/body_inertia/dof_armature changes, and ``set_const_fixed``
        is the cheapest (covers body_gravcomp changes).
    """
    fn = getattr(mjwarp, level.name)
    fn(self._wp_model, self._wp_data)

  def forward(self) -> None:
    mjwarp.forward(self.wp_model, self.wp_data)

  def step(self) -> None:
    with self.nan_guard.watch(self.data):
      mjwarp.step(self.wp_model, self.wp_data)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self._reset_mask.fill_(True)
    else:
      self._reset_mask.fill_(False)
      self._reset_mask[env_ids] = True

    mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_qd)

  def set_sensor_context(self, ctx: SensorContextCPU) -> None:
    """Wire a sensor context for camera/raycast sensing.

    The qd backend consumes :class:`mjlab.sensor.sensor_context_cpu.
    SensorContextCPU` (torch raycasts + the qd/GL camera backends), which
    the scene builds for every backend; sensing runs on torch tensors
    against the bridged state.
    """
    if not hasattr(ctx, "sense"):
      raise NotImplementedError(
        "The qd backend requires SensorContextCPU (from "
        "mjlab.sensor.sensor_context_cpu)."
      )
    self._sensor_context = ctx

  def sense(self) -> None:
    """Run raycast computation and camera rendering for this step."""
    if self._sensor_context is not None:
      self._sensor_context.sense()

  def close(self) -> None:
    """Release background resources (e.g. the async camera render thread)."""
    if self._sensor_context is not None and hasattr(self._sensor_context, "close"):
      self._sensor_context.close()


def make_simulation(
  num_envs: int,
  cfg: SimulationCfg,
  model: mujoco.MjModel | None = None,
  device: str = "cpu",
  *,
  spec: mujoco.MjSpec | None = None,
  variant_info: list[tuple[str, "VariantMetadata"]] | None = None,
) -> Simulation:
  """Construct the simulation backend selected by ``cfg.backend``.

  Returns a :class:`Simulation` (qd) or a
  :class:`mjlab.sim.classic.ClassicSimulation`; the two expose the same
  interface, so the return value is typed as ``Simulation``.
  """
  if cfg.backend == "classic":
    from mjlab.sim.classic import ClassicSimulation

    return cast(
      Simulation,
      ClassicSimulation(
        num_envs=num_envs,
        cfg=cfg,
        model=model,
        device=device,
        spec=spec,
        variant_info=variant_info,
      ),
    )
  return Simulation(
    num_envs=num_envs,
    cfg=cfg,
    model=model,
    device=device,
    spec=spec,
    variant_info=variant_info,
  )
