"""CPU raycast for the classic backend: GridPatternCfg height scans.

Torch-vectorized ray intersection with hfield surfaces and planes, matching
mujoco_warp's ``rays`` semantics (nearest hit across candidate geoms; -1
distance, zero normal and ray origin on miss).

The numeric core is compiled with ``torch.compile`` when available (the
pointwise chains are dispatch-bound in eager mode on CPU); a plain eager
fallback is used if compilation fails. Results are identical either way.

Scope (v1):
- hfield intersection supports rays that are vertical in the hfield's local
  frame (the height-scan use case; a clear ``NotImplementedError`` is raised
  otherwise). The surface is intersected exactly like mujoco_warp's
  ``ray_hfield``: two triangles per grid cell, split along the cell diagonal.
- planes support arbitrary directions (analytic, front-face only, bounded by
  ``geom_size`` like ``ray_plane``).
- other geom types (box, mesh, ...) do not participate; keep them out of the
  ray volume or use ``backend='warp'`` for full-scene raycasting.
- hfield data and the poses of geoms on jointless (static) bodies are
  snapshotted from the template model at construction; per-environment
  randomization of those is not picked up. Geoms on dynamic bodies use the
  per-environment poses every call.

The ray-preparation logic in :func:`compute_world_rays` and the
post-processing in :func:`finalize` mirror ``RayCastSensor.prepare_rays`` /
``postprocess_rays`` (kept in sync with those); they live here so the Warp
sensor class needs no backend branching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_from_matrix

_BIG = 1e30
_MJ_MINVAL = 1e-15
# Rays more than this far from vertical (in the hfield local frame) are rejected.
_VERTICAL_TOL = 0.999

# How a candidate geom's world pose is obtained each call.
# - "identity": static at the origin with identity rotation (no transform).
# - "constant": static elsewhere (constant transform baked at construction).
# - "dynamic": read per-environment geom_xpos/geom_xmat each call.
_PoseMode = Literal["identity", "constant", "dynamic"]


@dataclass
class _Hfield:
  geom_id: int
  body_id: int
  pose_mode: _PoseMode
  # Per-cell interleaved corner table: [nrow * ncol, 4] world-heights
  # (h00, h10, h01, h11) of each grid cell, scaled by size[2]. Edge cells
  # replicate their boundary corner so clamped gathers stay in bounds.
  corners: torch.Tensor
  nrow: int
  ncol: int
  inv_dx: float
  bx: float  # local x * inv_dx + bx == grid u coordinate
  inv_dy: float
  by: float
  dx: float
  dy: float
  # Baked constant transform for pose_mode == "constant".
  pos0: torch.Tensor | None  # [3]
  rot0: torch.Tensor | None  # [3, 3]


@dataclass
class _Plane:
  geom_id: int
  body_id: int
  pose_mode: _PoseMode
  sx: float  # geom_size x half-extent (<= 0: unbounded)
  sy: float
  pos0: torch.Tensor | None
  rot0: torch.Tensor | None


def _geom_invisible(mj_model: mujoco.MjModel, geom_id: int) -> bool:
  """Mirror mujoco_warp's invisible-geom exclusion (fully transparent)."""
  matid = int(mj_model.geom_matid[geom_id])
  if matid < 0:
    return float(mj_model.geom_rgba[geom_id][3]) == 0.0
  return float(mj_model.mat_rgba[matid][3]) == 0.0


def compute_world_rays(
  sensor: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Transform a sensor's local rays to world frame (pure torch).

  Mirrors ``RayCastSensor.prepare_rays`` (which also writes the same cached
  fields for the Warp path); the CPU path uses this variant to avoid the
  Warp array round-trip. The alignment-rotation math is reused from the
  sensor itself.

  Returns:
    (frame_pos [B, F, 3], frame_mat [B, F, 3, 3],
     world_origins [B, F*N, 3], world_dirs [B, F*N, 3]).
  """
  data = sensor._data
  local_offsets = sensor._local_offsets
  local_directions = sensor._local_directions

  pos_list: list[torch.Tensor] = []
  mat_list: list[torch.Tensor] = []
  for frame_type, obj_id, _ in sensor._frame_infos:
    if frame_type == "body":
      pos_list.append(data.xpos[:, obj_id])
      mat_list.append(data.xmat[:, obj_id].view(-1, 3, 3))
    elif frame_type == "site":
      pos_list.append(data.site_xpos[:, obj_id])
      mat_list.append(data.site_xmat[:, obj_id].view(-1, 3, 3))
    else:  # geom
      pos_list.append(data.geom_xpos[:, obj_id])
      mat_list.append(data.geom_xmat[:, obj_id].view(-1, 3, 3))

  frame_pos = torch.stack(pos_list, dim=1)  # [B, F, 3]
  frame_mat = torch.stack(mat_list, dim=1)  # [B, F, 3, 3]

  B, F = frame_pos.shape[:2]
  N = local_offsets.shape[0]

  rot_mat = sensor._compute_alignment_rotation(frame_mat.reshape(B * F, 3, 3)).reshape(
    B, F, 3, 3
  )

  world_offsets = torch.einsum("bfij,nj->bfni", rot_mat, local_offsets)
  world_origins = frame_pos[:, :, None, :] + world_offsets
  world_rays = torch.einsum("bfij,nj->bfni", rot_mat, local_directions)

  world_origins_flat = world_origins.reshape(B, F * N, 3)
  world_rays_flat = world_rays.reshape(B, F * N, 3)

  # Cache for finalize() and debug_vis().
  sensor._cached_world_origins = world_origins_flat
  sensor._cached_world_rays = world_rays_flat
  sensor._cached_frame_pos = frame_pos  # [B, F, 3]
  sensor._cached_frame_mat = frame_mat  # [B, F, 3, 3]

  return frame_pos, frame_mat, world_origins_flat, world_rays_flat


def finalize(
  sensor: Any,
  distances: torch.Tensor,
  normals_w: torch.Tensor,
  world_origins: torch.Tensor,
  world_dirs: torch.Tensor,
) -> None:
  """Post-process CPU raycast outputs onto a ``RayCastSensor``.

  Mirrors ``RayCastSensor.postprocess_rays`` semantics: miss rays carry -1
  distance, zero normals and the ray origin as hit position.
  """
  frame_pos = sensor._cached_frame_pos
  frame_mat = sensor._cached_frame_mat
  B, F = frame_pos.shape[:2]

  # ``origin + ray * max(distance, 0)`` collapses miss rays to ``origin``
  # (clamped distance is 0) without any branching.
  clamped = distances.clamp(min=0.0)
  sensor._hit_pos_w = world_origins + world_dirs * clamped.unsqueeze(-1)
  sensor._distances = distances
  sensor._normals_w = normals_w

  # All frames: [B, F, 3] / [B, F, 4]; first frame for backward compat.
  sensor._frame_pos_w = frame_pos
  sensor._frame_quat_w = quat_from_matrix(frame_mat.reshape(B * F, 3, 3)).reshape(
    B, F, 4
  )
  sensor._pos_w = frame_pos[:, 0]
  sensor._quat_w = sensor._frame_quat_w[:, 0]

  # Force a recompute on the next ``.data`` access (see
  # RayCastSensor.postprocess_rays for why, #998).
  sensor._invalidate_cache()


class CpuRaycastContext:
  """Per-sensor CPU raycast state: candidate hfield/plane geoms of one model.

  Candidate selection mirrors mujoco_warp's ``_ray_eliminate``: geom group
  inclusion, per-ray body exclusion, static geoms allowed, and invisible
  (fully transparent) geoms dropped.
  """

  def __init__(self, mj_model: mujoco.MjModel, sensor: Any, data: Any) -> None:
    self._mj_model = mj_model
    self._data = data
    self._max_distance = float(sensor.cfg.max_distance)
    self._rays_per_frame = sensor.num_rays_per_frame

    # Template-model world poses, for classifying static geoms.
    scratch = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, scratch)

    # Per-ray body exclusion, mirroring RayCastSensor.initialize.
    rays_per_frame = sensor.num_rays_per_frame
    if sensor.cfg.exclude_parent_body:
      body_excludes = [-1] * sensor.num_rays
      for i, (_, _, body_id) in enumerate(sensor._frame_infos):
        body_excludes[i * rays_per_frame : (i + 1) * rays_per_frame] = [body_id] * (
          rays_per_frame
        )
    else:
      body_excludes = [-1] * sensor.num_rays
    self._body_exclude = torch.tensor(body_excludes, dtype=torch.long)

    groups = sensor.cfg.include_geom_groups
    include_all = groups is None
    include = set(range(mujoco.mjNGROUP)) if include_all else set(groups)

    self._hfields: list[_Hfield] = []
    self._planes: list[_Plane] = []
    for g in range(mj_model.ngeom):
      if not include_all:
        group = int(np.clip(int(mj_model.geom_group[g]), 0, mujoco.mjNGROUP - 1))
        if group not in include:
          continue
      if _geom_invisible(mj_model, g):
        continue
      body_id = int(mj_model.geom_bodyid[g])
      pose_mode, pos0, rot0 = self._classify_pose(mj_model, scratch, g, body_id)
      geom_type = int(mj_model.geom_type[g])
      if geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD):
        self._hfields.append(
          self._build_hfield(mj_model, g, body_id, pose_mode, pos0, rot0)
        )
      elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        size = np.asarray(mj_model.geom_size[g])
        self._planes.append(
          _Plane(
            geom_id=g,
            body_id=body_id,
            pose_mode=pose_mode,
            sx=float(size[0]),
            sy=float(size[1]),
            pos0=pos0,
            rot0=rot0,
          )
        )

    self._height_scan_fn = self._make_height_scan_fn()

  def _classify_pose(
    self,
    mj_model: mujoco.MjModel,
    scratch: mujoco.MjData,
    geom_id: int,
    body_id: int,
  ) -> tuple[_PoseMode, torch.Tensor | None, torch.Tensor | None]:
    """Classify how the geom's world pose is obtained (see _PoseMode)."""
    # A geom on a body with no joints anywhere along its parent chain never
    # moves: its world pose is the template constant.
    b = body_id
    while b != 0:
      if int(mj_model.body_jntnum[b]) > 0:
        return "dynamic", None, None
      b = int(mj_model.body_parentid[b])
    pos = np.asarray(scratch.geom_xpos[geom_id])
    rot = np.asarray(scratch.geom_xmat[geom_id]).reshape(3, 3)
    if not np.any(pos != 0.0) and np.allclose(rot, np.eye(3)):
      return "identity", None, None
    return (
      "constant",
      torch.as_tensor(np.ascontiguousarray(pos), dtype=torch.float32),
      torch.as_tensor(np.ascontiguousarray(rot), dtype=torch.float32),
    )

  def _build_hfield(
    self,
    mj_model: mujoco.MjModel,
    geom_id: int,
    body_id: int,
    pose_mode: _PoseMode,
    pos0: torch.Tensor | None,
    rot0: torch.Tensor | None,
  ) -> _Hfield:
    hid = int(mj_model.geom_dataid[geom_id])
    nrow = int(mj_model.hfield_nrow[hid])
    ncol = int(mj_model.hfield_ncol[hid])
    adr = int(mj_model.hfield_adr[hid])
    size = np.asarray(mj_model.hfield_size[hid])
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    raw = np.array(mj_model.hfield_data[adr : adr + nrow * ncol]).reshape(nrow, ncol)
    f = torch.as_tensor(np.ascontiguousarray(raw), dtype=torch.float32) * sz
    # Per-cell corner table (h00, h10, h01, h11); edge rows/cols replicate so
    # clamped corner gathers never leave the table.
    r1 = torch.clamp(torch.arange(nrow) + 1, max=nrow - 1)
    c1 = torch.clamp(torch.arange(ncol) + 1, max=ncol - 1)
    h00 = f
    h10 = f[:, c1]
    h01 = f[r1]
    h11 = f[r1][:, c1]
    corners = torch.stack(
      [h00.reshape(-1), h10.reshape(-1), h01.reshape(-1), h11.reshape(-1)],
      dim=-1,
    ).contiguous()
    dx = 2.0 * sx / (ncol - 1)
    dy = 2.0 * sy / (nrow - 1)
    return _Hfield(
      geom_id=geom_id,
      body_id=body_id,
      pose_mode=pose_mode,
      corners=corners,
      nrow=nrow,
      ncol=ncol,
      inv_dx=1.0 / dx,
      bx=sx / dx,
      inv_dy=1.0 / dy,
      by=sy / dy,
      dx=dx,
      dy=dy,
      pos0=pos0,
      rot0=rot0,
    )

  # Numeric core. Each candidate contributes a hit-distance tensor (+inf on
  # miss) and world-frame normal components; the nearest hit wins. Written as
  # one pure-torch function so torch.compile can fuse the pointwise chains
  # (eager mode is dispatch-bound at these tensor sizes).

  def _make_height_scan_fn(self):
    """Build the (compiled when possible) height-scan closure for this ctx."""
    hfields = self._hfields
    planes = self._planes
    max_distance = self._max_distance

    def core(
      origins: torch.Tensor,
      directions: torch.Tensor,
      inv_ldz0: torch.Tensor,
    ):
      dtype = origins.dtype
      tmin = torch.full(origins.shape[:2], _BIG, dtype=dtype)
      normals = torch.zeros(*origins.shape[:2], 3, dtype=dtype)

      for hf in hfields:
        if hf.pose_mode == "identity":
          lo = origins
          ld = directions
          # Grid rays share one direction per frame, so the local-frame ldz
          # is constant across a frame's rays: multiply by the per-frame
          # reciprocal ([B, 1] for single-frame sensors, [B, N] otherwise)
          # instead of dividing per ray.
          inv_ldz = inv_ldz0
        elif hf.pose_mode == "constant":
          lo = (origins - hf.pos0) @ hf.rot0
          ld = directions @ hf.rot0
          inv_ldz = 1.0 / ld[..., 2]
        else:
          xpos = self._data.geom_xpos[:, hf.geom_id]
          xmat = self._data.geom_xmat[:, hf.geom_id]
          lo = (origins - xpos.unsqueeze(1)) @ xmat
          ld = directions @ xmat
          inv_ldz = 1.0 / ld[..., 2]

        u = lo[..., 0] * hf.inv_dx + hf.bx
        v = lo[..., 1] * hf.inv_dy + hf.by
        uc = u.clamp(0, hf.ncol - 1)
        iu = uc.eq(u)
        vc = v.clamp(0, hf.nrow - 1)
        iv = vc.eq(v)
        cf = uc.floor()
        fu = uc.sub_(cf)
        rf = vc.floor()
        fv = vc.sub_(rf)
        r = rf.to(torch.long)
        c = cf.to(torch.long)
        # One gather per ray from the interleaved per-cell corner table: the
        # four corners share a cache line. Edge cells replicate their boundary
        # corners, so the clamped cell index always lands in the table
        # (out-of-footprint rays are masked by iu/iv before use).
        lin = r.mul_(hf.ncol).add_(c).mul_(4)
        table = hf.corners.reshape(-1)
        h00 = table.take(lin)
        h10 = table.take(lin + 1)
        h01 = table.take(lin + 2)
        h11 = table.take(lin + 3)

        # Two triangles per cell, split along the (0,0)-(1,1) diagonal,
        # exactly like mujoco_warp's ray_hfield.
        t1 = h10 - h00
        t2 = h11 - h10
        t3 = h01 - h00
        t4 = h11 - h01
        tri1 = fu >= fv
        z1 = h00 + fu * t1 + fv * t2
        z2 = h00 + fv * t3 + fu * t4
        z = torch.where(tri1, z1, z2)

        # Hit distance along the ray: lo + t * ld reaches the surface when
        # lo_z + t * ldz == z.
        dist = (z - lo[..., 2]) * inv_ldz
        better = iu & iv & (dist >= 0.0) & (dist <= max_distance) & (dist < tmin)
        tmin = torch.where(better, dist, tmin)

        # Exact triangle normals (up like ray_hfield's cross products).
        nx1 = t1 * (-hf.dy)
        b1 = t1 + t2
        ny1 = (t1 - b1) * hf.dx
        nx2 = (t3 - b1) * hf.dy
        ny2 = t3 * (-hf.dx)
        nx = torch.where(tri1, nx1, nx2)
        ny = torch.where(tri1, ny1, ny2)
        q = nx * nx + ny * ny + hf.dx * hf.dx * hf.dy * hf.dy
        inv = q.rsqrt()
        nx = nx * inv
        ny = ny * inv
        nz = inv * (hf.dx * hf.dy)
        if hf.pose_mode != "identity":
          n_local = torch.stack([nx, ny, nz], dim=-1)
          if hf.pose_mode == "constant":
            n_world = n_local @ hf.rot0
          else:
            n_world = torch.einsum("bij,bnj->bni", xmat, n_local)
          nx, ny, nz = n_world[..., 0], n_world[..., 1], n_world[..., 2]

        w = better.unsqueeze(-1)
        normals = torch.where(w, torch.stack([nx, ny, nz], dim=-1), normals)

      for plane in planes:
        if plane.pose_mode == "identity":
          lo = origins
          ld = directions
          # Grid directions are uniform across a frame's rays: the front-face
          # check and the reciprocal collapse to per-(env, frame) values.
          ok = torch.broadcast_to(inv_ldz0 < 0.0, origins.shape[:2])
          x = -lo[..., 2] * inv_ldz0
        elif plane.pose_mode == "constant":
          lo = (origins - plane.pos0) @ plane.rot0
          ld = directions @ plane.rot0
          denom = ld[..., 2]
          ok = denom < -_MJ_MINVAL
          x = -lo[..., 2] / denom.clamp(max=-_MJ_MINVAL)
        else:
          xpos = self._data.geom_xpos[:, plane.geom_id]
          xmat = self._data.geom_xmat[:, plane.geom_id]
          lo = (origins - xpos.unsqueeze(1)) @ xmat
          ld = directions @ xmat
          denom = ld[..., 2]
          ok = denom < -_MJ_MINVAL
          x = -lo[..., 2] / denom.clamp(max=-_MJ_MINVAL)
        ok = ok & (x >= 0.0) & (x <= max_distance)
        u = lo[..., 0] + x * ld[..., 0]
        v = lo[..., 1] + x * ld[..., 1]
        if plane.sx > 0.0:
          ok = ok & (u.abs() <= plane.sx)
        if plane.sy > 0.0:
          ok = ok & (v.abs() <= plane.sy)
        better = ok & (x < tmin)
        tmin = torch.where(better, x, tmin)

        if plane.pose_mode == "identity":
          zero = lo[..., 2] * 0.0
          nx, ny, nz = zero, zero, zero + 1.0
        elif plane.pose_mode == "constant":
          nx = torch.full(lo.shape[:2], float(plane.rot0[0, 2]))
          ny = torch.full(lo.shape[:2], float(plane.rot0[1, 2]))
          nz = torch.full(lo.shape[:2], float(plane.rot0[2, 2]))
        else:
          xmat = self._data.geom_xmat[:, plane.geom_id]
          nx = xmat[:, 0, 2].unsqueeze(1).expand(x.shape)
          ny = xmat[:, 1, 2].unsqueeze(1).expand(x.shape)
          nz = xmat[:, 2, 2].unsqueeze(1).expand(x.shape)
        w = better.unsqueeze(-1)
        normals = torch.where(w, torch.stack([nx, ny, nz], dim=-1), normals)

      # Misses report -1; beyond-range hits were excluded in the merges
      # (mirroring the Warp post-processing clip).
      distances = torch.where(tmin < _BIG, tmin, -1.0)
      return distances, normals

    try:
      return torch.compile(core, dynamic=False)
    except Exception:
      return core

  def height_scan(
    self,
    origins: torch.Tensor,
    directions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect rays with candidate hfields and planes.

    Args:
      origins: [B, N, 3] world-frame ray origins.
      directions: [B, N, 3] world-frame ray directions (normalized).

    Returns:
      (distances [B, N], normals_w [B, N, 3]); miss rays carry -1 and 0.

    Raises:
      NotImplementedError: If hfield candidates exist and any ray is not
        vertical in the hfield's local frame.
    """
    B, N, _ = directions.shape
    F = max(N // self._rays_per_frame, 1)
    # GridPattern rays share one direction per frame: collapse to one
    # direction per (env, frame) for the verticality check and reciprocal.
    dirs0 = directions.view(B, F, N // F, 3)[:, :, 0]
    if self._hfields and not self._rays_vertical(dirs0):
      raise NotImplementedError(
        "CPU raycast hfield intersection supports rays that are vertical in "
        "the hfield's local frame only (GridPatternCfg height scans); use "
        "backend='warp' for general ray directions."
      )
    inv = torch.reciprocal(dirs0[..., 2])
    if F == 1:
      inv_ldz0 = inv.reshape(B, 1)
    else:
      inv_ldz0 = torch.repeat_interleave(inv, N // F, dim=1)
    # The fused kernel is small per element: a reduced intra-op pool avoids
    # per-group thread syncs that cost more than the arithmetic (measured on
    # M1 Pro); the toggle is effectively free and restored in ``finally``.
    prev = torch.get_num_threads()
    torch.set_num_threads(min(prev, 4))
    try:
      return self._height_scan_fn(origins, directions, inv_ldz0)
    finally:
      torch.set_num_threads(prev)

  def _rays_vertical(self, dirs0: torch.Tensor) -> bool:
    """Whether every (env, frame) direction is vertical in each hfield's
    local frame. dirs0: [B, F, 3] collapsed per-frame directions."""
    for hf in self._hfields:
      if hf.pose_mode == "identity":
        ldz = dirs0[..., 2]
      elif hf.pose_mode == "constant":
        ldz = (dirs0 @ hf.rot0)[..., 2]
      else:
        xmat = self._data.geom_xmat[:, hf.geom_id]
        ldz = torch.einsum("bfi,bij->bfj", dirs0, xmat)[..., 2]
      if float(ldz.abs().min()) < _VERTICAL_TOL:
        return False
    return True
