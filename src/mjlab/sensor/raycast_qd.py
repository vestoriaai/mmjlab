"""Quadrants (qd) raycast backend for the classic backend (stage 1B).

Wraps the validated stage-1A intersection library (``qd_render_poc``) as a
per-sensor context with the same semantics as
:class:`mjlab.sensor.raycast_core.RaycastCoreContext`: ``closest_hit`` returns
(distances, normals_w, hit_pos_w) with ``distances=-1`` / ``normals=0`` /
``hit_pos=origin`` on miss.

Design (probed on Metal, see docs/results/worklog-1B.md):

- **Scene arrays stay resident on the device**; rays launch in batches. The
  production path fuses ray generation into the kernel
  (``ray_scene_pattern``): the host only uploads one packed per-frame pose
  array (W*F*12 f32) and one packed per-env geom pose array (W*G*12 f32),
  because every ``qd`` array upload carries a ~0.2 ms fixed command-buffer
  cost on Metal — uploading explicit rays (W*R*6 f32) is the dominant e2e
  cost at the lidar gate load.
- **Outputs pass through DLPack**: ``qd.ndarray.to_dlpack()`` +
  ``torch.from_dlpack`` alias the device buffer as an mps/cuda tensor; one
  ``.cpu()`` materializes the interleaved (N, 4) [dist, nx, ny, nz] block
  (~4x faster than ``to_numpy`` on Metal). Strided views into that block are
  returned as the distance/normal tensors.
- **Exact cross-environment dedup**: when all envs share bit-identical frame
  and geom poses, one world is launched and outputs tiled.
- ``quadrants`` / ``qd_render_poc`` are **optional dependencies**: every
  entry point is import-guarded and the context construction falls back to
  the torch core (see ``sensor_context_cpu``).

Scope notes:
- Geom candidate selection mirrors the torch core (geom group inclusion,
  fully-transparent geoms dropped, ``flg_static=True``); excluded geoms are
  patched out of the scene arrays at construction.
- Poses of geoms on jointless bodies are the template snapshot (identical in
  every env), matching :class:`~mjlab.sensor.raycast_cpu.CpuRaycastContext`;
  dynamic geoms read per-environment poses from the data bridge each call.
  Fully static scenes upload poses once.
- Mesh normals face against the ray direction (1A convention); for rays
  launched from outside closed meshes this equals the mj_ray / winding
  normal. Inside-origin hits keep the facing convention (recorded as normal
  sign flips in alignment tests).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab.sensor import raycast_cpu
from mjlab.sensor.raycast_cpu import _geom_invisible

_qd_modules: dict[str, Any] | None = None
_qd_arch: str | None = None
_qd_init_failed = False


def is_available() -> bool:
  """Whether the qd backend *could* be used (imports only; no device init)."""
  try:
    import quadrants  # noqa: F401
    import qd_render_poc  # noqa: F401
  except ImportError:
    return False
  return True


def _modules() -> dict[str, Any]:
  global _qd_modules
  if _qd_modules is None:
    import quadrants
    import qd_render_poc
    from qd_render_poc import bvh, kernels, raycast, scene

    _qd_modules = {
      "qd": quadrants,
      "bvh": bvh,
      "kernels": kernels,
      "raycast": raycast,
      "scene": scene,
    }
  return _qd_modules


def qd_arch() -> str | None:
  """The initialized qd arch, or None if qd is unavailable/failed to init."""
  if _qd_init_failed:
    return None
  _ensure_init()
  return _qd_arch


def _ensure_init() -> bool:
  """Initialize quadrants once per process (metal > cuda > vulkan > cpu)."""
  global _qd_arch, _qd_init_failed
  if _qd_arch is not None:
    return True
  if _qd_init_failed:
    return False
  if not is_available():
    _qd_init_failed = True
    return False
  qd = _modules()["qd"]
  override = os.environ.get("MJLAB_RAYCAST_QD_ARCH")
  candidates = (
    [override] if override else (["metal"] if sys.platform == "darwin" else [])
  ) + ["cuda", "vulkan", "cpu"]
  for arch in candidates:
    try:
      qd.init(arch=getattr(qd, arch))
      _qd_arch = arch
      return True
    except Exception:
      continue
  _qd_init_failed = True
  return False


def scene_has_qd_only_types(mj_model: mujoco.MjModel) -> bool:
  """Whether the scene contains geom types the torch core ignores but qd
  supports (sphere/capsule/ellipsoid/cylinder/mesh).

  Drives the ``raycast_backend="auto"`` policy: ``auto`` keeps the proven
  torch fast path for plane/hfield scenes (where it is measurably faster at
  the crossover-relevant loads and pins the hfield non-vertical raise), and
  selects qd exactly where the torch core would silently ignore geometry.
  """
  supported = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_MESH),
  }
  return any(int(t) in supported for t in mj_model.geom_type)


def _has_dynamic_geom(mj_model: mujoco.MjModel) -> bool:
  """Whether any geom sits on a body with joints anywhere up its chain."""
  body_jntnum = mj_model.body_jntnum
  parent = mj_model.body_parentid
  for bid in range(mj_model.nbody):
    b = bid
    while b != 0:
      if int(body_jntnum[b]) > 0:
        return True
      b = int(parent[b])
  return False


class QdRaycastContext:
  """Per-sensor qd raycast context (classic backend).

  Same construction signature and ``closest_hit`` contract as
  :class:`~mjlab.sensor.raycast_core.RaycastCoreContext`.
  """

  def __init__(self, mj_model: mujoco.MjModel, sensor: Any, data: Any) -> None:
    mods = _modules()
    if not _ensure_init():
      raise RuntimeError("quadrants could not be initialized on this machine")
    self._mj_model = mj_model
    self._data = data
    self._mods = mods
    self._max_distance = float(sensor.cfg.max_distance)
    self._rays_per_frame = sensor.num_rays_per_frame
    self._num_frames = sensor.num_frames
    self._num_rays = sensor.num_rays
    self._nworld = int(data.nworld)

    # Per-frame body exclusion (mirrors RayCastSensor.initialize).
    if sensor.cfg.exclude_parent_body:
      self._frame_body_exclude = [int(bid) for _, _, bid in sensor._frame_infos]
    else:
      self._frame_body_exclude = [-1] * self._num_frames

    # Candidate selection, mirroring RaycastCoreContext: geom group
    # inclusion + fully-transparent geoms dropped. Excluded geoms are
    # patched to geom_type -1 (never dispatched by the kernels), so the
    # runtime geomgroup filter stays off.
    groups = sensor.cfg.include_geom_groups
    include_all = groups is None
    include = set(range(mujoco.mjNGROUP)) if include_all else set(groups)
    keep = np.ones(mj_model.ngeom, dtype=bool)
    for g in range(mj_model.ngeom):
      if not include_all:
        group = int(np.clip(int(mj_model.geom_group[g]), 0, mujoco.mjNGROUP - 1))
        if group not in include:
          keep[g] = False
          continue
      if _geom_invisible(mj_model, g):
        keep[g] = False

    scratch = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, scratch)
    scene = mods["scene"].extract_scene(mj_model, scratch)
    scene.geom_type = np.where(keep, scene.geom_type, np.int32(-1)).astype(np.int32)

    # Mesh candidates that survive selection (compat surface for tests that
    # introspect the torch core's `mesh_colliders`).
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    self.mesh_colliders = [
      g for g in range(mj_model.ngeom)
      if keep[g] and int(mj_model.geom_type[g]) == mesh_type
    ]

    self._its = mods["raycast"].QdIntersector(scene, nworld=self._nworld)
    self._scene = scene
    self._ngeom = scene.ngeom
    self._static_scene = not _has_dynamic_geom(mj_model)

    qd = mods["qd"]
    # Pattern device arrays (static per sensor config).
    offsets = sensor._local_offsets
    dirs = sensor._local_directions
    self._q_offsets = qd.ndarray(qd.f32, tuple(offsets.shape))
    self._q_offsets.from_numpy(
      np.ascontiguousarray(offsets.detach().numpy(), dtype=np.float32))
    self._q_dirs0 = qd.ndarray(qd.f32, tuple(dirs.shape))
    self._q_dirs0.from_numpy(
      np.ascontiguousarray(dirs.detach().numpy(), dtype=np.float32))
    self._q_geomgroup = qd.ndarray(qd.i32, (6,))
    self._q_geomgroup.from_numpy(np.full(6, 1, dtype=np.int32))
    self._q_flags = qd.ndarray(qd.i32, (3,))
    self._q_flags.from_numpy(np.array([0, 1, self._rays_per_frame], dtype=np.int32))
    self._q_bex = qd.ndarray(qd.i32, (self._num_frames,))
    self._q_bex.from_numpy(np.array(self._frame_body_exclude, dtype=np.int32))
    self._q_max_dist = qd.ndarray(qd.f32, (1,))
    self._q_max_dist.from_numpy(np.array([self._max_distance], dtype=np.float32))

    # Per-worlds-key device buffers; _host_cache tracks the last-uploaded
    # host copies so redundant uploads are skipped.
    self._q_frame_pose: dict[int, Any] = {}
    self._q_pose: dict[int, Any] = {}
    self._q_out: dict[tuple[int, int], Any] = {}
    self._q_stack: dict[tuple[int, int], Any] = {}
    # Last-uploaded host copies per (kind, worlds): each upload carries a
    # ~0.2ms fixed cost on Metal, so skip no-op uploads when the host arrays
    # are unchanged (the common case for repeated reads of one sim state).
    self._host_cache: dict[tuple[str, int], np.ndarray] = {}

  def _upload_if_changed(
    self,
    kind_map: dict[int, Any],
    worlds: int,
    qd_array,
    host: np.ndarray,
    force: bool = False,
  ) -> None:
    key = (id(kind_map), worlds)
    cached = self._host_cache.get(key)
    if force or cached is None or cached.shape != host.shape or not bool(
      (cached == host).all()
    ):
      qd_array.from_numpy(np.ascontiguousarray(host, dtype=np.float32))
      self._host_cache[key] = host

  # ------------------------------------------------------------------
  # Device buffer helpers.
  # ------------------------------------------------------------------

  def _buffers(self, worlds: int):
    qd = self._mods["qd"]
    key = (worlds, self._num_rays)
    if key not in self._q_out:
      from qd_render_poc.bvh import STACK_DEPTH

      self._q_out[key] = qd.ndarray(qd.f32, (worlds * self._num_rays, 4))
      self._q_stack[key] = qd.ndarray(
        qd.i32, (worlds * self._num_rays, STACK_DEPTH))
      self._q_frame_pose[worlds] = qd.ndarray(
        qd.f32, (worlds * self._num_frames, 12))
      self._q_pose[worlds] = qd.ndarray(qd.f32, (worlds * self._ngeom, 12))
    return (
      self._q_frame_pose[worlds],
      self._q_pose[worlds],
      self._q_out[key],
      self._q_stack[key],
    )

  def _static_pose(self, worlds: int) -> np.ndarray:
    """Template poses tiled per env, packed as (W*G, 12)."""
    pose = np.zeros((worlds * self._ngeom, 12), dtype=np.float32)
    pv = pose.reshape(worlds, self._ngeom, 12)
    pv[:, :, 0:3] = self._scene.geom_xpos.reshape(self._ngeom, 3).astype(
      np.float32)[None]
    pv[:, :, 3:12] = self._scene.geom_xmat.reshape(self._ngeom, 9).astype(
      np.float32)[None]
    return np.ascontiguousarray(pose)

  def _geom_pose_array(self, worlds: int) -> np.ndarray | None:
    """Packed per-env geom poses [xpos(3), xmat(9)] as one (W*G, 12) array.

    Returns None for fully static scenes at full batch (the caller uploads
    the tiled template once; repeated identical uploads are skipped by the
    host cache).
    """
    if self._static_scene:
      return None if worlds == self._nworld else self._static_pose(worlds)
    data = self._data
    xpos = data.geom_xpos
    xmat = data.geom_xmat
    if xpos.dim() == 2:  # single env without a world lead dim
      xpos = xpos.unsqueeze(0)
      xmat = xmat.unsqueeze(0)
    packed = torch.cat(
      (xpos.reshape(self._nworld, self._ngeom, 3),
       xmat.reshape(self._nworld, self._ngeom, 9)),
      dim=-1,
    )
    if worlds != self._nworld:
      packed = packed[:worlds]
    return np.ascontiguousarray(packed.detach().reshape(-1, 12).numpy())

  # ------------------------------------------------------------------
  # Production path: fused raygen + intersect (pattern native).
  # ------------------------------------------------------------------

  def closest_hit_pattern(
    self,
    sensor: Any,
    dedup: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect the sensor's current ray pattern on the qd device.

    Requires :func:`~mjlab.sensor.raycast_cpu.compute_world_rays` to have
    run for ``sensor`` (frame pose caches). Returns ``(distances, normals_w)``
    with RayCastData miss semantics (``-1`` / ``0``); hit position is derived
    by ``raycast_cpu.finalize`` from the cached world rays.

    ``dedup`` enables exact cross-environment deduplication: when every env
    equals env 0 translated by one constant offset (mjlab's grid layout) and
    rotations match bitwise, one world is launched and the outputs tiled
    (translation-invariant up to f32 rounding, ~1e-7 rel).
    """
    qd = self._mods["qd"]
    frame_pos = sensor._cached_frame_pos  # [B, F, 3]
    frame_mat = sensor._cached_frame_mat  # [B, F, 3, 3]
    B, F = frame_pos.shape[:2]
    assert B == self._nworld and F == self._num_frames

    rot = sensor._compute_alignment_rotation(
      frame_mat.reshape(B * F, 3, 3)
    ).reshape(B, F, 3, 3)

    worlds = B
    if dedup and B > 1:
      if self._translation_equivalent(frame_pos, rot):
        worlds = 1

    fp32 = frame_pos.detach().reshape(B * F, 3).numpy().astype(np.float32)
    fr32 = rot.detach().reshape(B * F, 9).numpy().astype(np.float32)
    if worlds == B:
      frame_pose = np.ascontiguousarray(np.concatenate([fp32, fr32], axis=1))
    else:  # dedup: env 0 only
      frame_pose = np.ascontiguousarray(
        np.concatenate([fp32[:F], fr32[:F]], axis=1))

    qd_frame_pose, qd_pose, qd_out, qd_stack = self._buffers(worlds)
    self._upload_if_changed(self._q_frame_pose, worlds, qd_frame_pose, frame_pose)

    pose = self._geom_pose_array(worlds)
    if pose is None:  # static scene, full batch: device copy already current
      pose = self._static_pose(worlds)
    self._upload_if_changed(self._q_pose, worlds, qd_pose, pose)

    self._mods["kernels"].ray_scene_pattern(
      qd_frame_pose, self._q_offsets, self._q_dirs0,
      self._its.q_geom_type, self._its.q_geom_size, self._its.q_geom_group,
      self._its.q_geom_bodyid, self._its.q_body_weldid, qd_pose,
      self._q_geomgroup, self._q_flags, self._q_bex, self._q_max_dist,
      self._its.q_mesh_root_all, self._its.q_tri, self._its.q_bmin,
      self._its.q_bmax, self._its.q_left, self._its.q_right,
      self._its.q_first, self._its.q_count,
      self._its.q_hf_data, self._its.q_hf_adr, self._its.q_hf_nrow,
      self._its.q_hf_ncol, self._its.q_hf_size,
      qd_out, qd_stack,
    )
    qd.sync()
    distances, normals = self._read_out(qd_out, worlds)
    if worlds != B:  # dedup: tile env 0's outputs back to the full batch
      distances = distances.repeat(B, 1)
      normals = normals.repeat(B, 1, 1)
    return distances, normals

  def _translation_equivalent(self, frame_pos: torch.Tensor, rot: torch.Tensor) -> bool:
    """Whether every env equals env 0 translated by a constant offset.

    mjlab lays environments out on a grid: entities keep identical local
    poses, so env w's frame poses and geom poses are env 0's shifted by one
    per-env offset (until robot states diverge). Intersections are then
    identical per env (up to f32 rounding of the shifted operands, ~1e-7
    rel), so a single-world launch can serve the whole batch. Bitwise
    equality of the *differences* is the check (shifting and unshifting in
    f32 is not lossless).
    """
    if not bool((rot == rot[0:1]).all()):
      return False
    fd = frame_pos[:, 0, :] - frame_pos[0, 0, :]  # [B, 3]
    if not bool(((frame_pos - frame_pos[0:1]) == fd[:, None, :]).all()):
      return False
    if self._static_scene:
      # 静态几何逐 env 相同（不随 env 平移）→ 仅当帧也逐位相同才可去重。
      return not bool(fd.any())
    data = self._data
    xpos, xmat = data.geom_xpos, data.geom_xmat
    if xpos.dim() != 3:
      return True
    gd = xpos[:, 0, :] - xpos[0, 0, :]  # [B, 3] using geom 0 as the anchor
    if not bool(((xpos - xpos[0:1]) == gd[:, None, :]).all()):
      return False
    if not bool((gd == fd).all()):
      return False  # scene shift must equal the frame shift (static geoms etc.)
    return bool((xmat == xmat[0:1]).all())

  def _read_out(self, qd_out, worlds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Device (W*N,4) block -> (distances [W,N], normals [W,N,3]) CPU views."""
    block = self._wrap_torch(qd_out).cpu()
    N = self._num_rays
    flat = block.reshape(worlds * N, 4)
    distances = torch.as_strided(flat, (worlds, N), (N * 4, 4))
    normals = torch.as_strided(flat, (worlds, N, 3), (N * 4, 4, 1), storage_offset=1)
    return distances, normals

  @staticmethod
  def _wrap_torch(qd_array):
    """DLPack pass-through with a to_numpy fallback (non-DLPack archs)."""
    try:
      return torch.from_dlpack(qd_array.to_dlpack())
    except Exception:
      return torch.from_numpy(qd_array.to_numpy())

  # ------------------------------------------------------------------
  # End-to-end sense (production wiring).
  # ------------------------------------------------------------------

  def sense(self, sensor: Any) -> None:
    """Full sense pass for one sensor: ray prep, intersect, finalize."""
    _, _, origins, directions = raycast_cpu.compute_world_rays(sensor)
    distances, normals_w = self.closest_hit_pattern(sensor)
    raycast_cpu.finalize(sensor, distances, normals_w, origins, directions)

  # ------------------------------------------------------------------
  # Contract path: explicit world-frame rays (same signature as the torch
  # core's RaycastCoreContext.closest_hit).
  # ------------------------------------------------------------------

  def closest_hit(
    self,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """「几何场景 → 最近命中」 on explicit rays.

    Args:
      rays_o: [B, N, 3] world-frame ray origins.
      rays_d: [B, N, 3] world-frame ray directions (unit).

    Returns:
      (distances [B, N], normals_w [B, N, 3], hit_pos_w [B, N, 3]); miss
      rays carry distances=-1, normals=0 and hit_pos=origin.
    """
    qd = self._mods["qd"]
    B, N, _ = rays_o.shape
    F = self._num_frames
    R = N // F
    assert R * F == N
    o = np.ascontiguousarray(rays_o.detach().double().numpy())
    d = np.ascontiguousarray(rays_d.detach().double().numpy())
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    distances = np.full((B, N), -1.0)
    normals = np.zeros((B, N, 3))
    # One launch per distinct frame body exclusion (per-ray exclusion is
    # per-frame repeated in mjlab).
    for f in range(F):
      sl = slice(f * R, (f + 1) * R)
      dist_f, _, norm_f = self._its.closest_hit_scene(
        o[:, sl], d[:, sl],
        geomgroup=None, flg_static=True,
        bodyexclude=self._frame_body_exclude[f],
      )
      distances[:, sl] = dist_f
      normals[:, sl] = norm_f
    qd.sync()

    distances_t = torch.from_numpy(distances.astype(np.float32))
    distances_t = distances_t.masked_fill(distances_t > self._max_distance, -1.0)
    hit = distances_t >= 0
    normals_t = torch.from_numpy(normals.astype(np.float32)) * hit.unsqueeze(-1)
    clamped = distances_t.clamp(min=0.0)
    hit_pos = (
      rays_o.reshape(B, N, 3) + rays_d.reshape(B, N, 3) * clamped.unsqueeze(-1)
    )
    return distances_t, normals_t, hit_pos
