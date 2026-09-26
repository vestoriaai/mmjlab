"""Batched camera rendering for the classic backend via the mujoco_warp
renderer on CPU (stage 0 of the rendering roadmap).

:class:`MjwarpCameraContext` mirrors :class:`mjlab.sensor.camera_cpu.
CpuCameraContext`'s surface (``render`` / ``get_rgb`` / ``get_depth`` /
``_render_states``) but renders all environments at once through a shadow
mujoco_warp Model/Data pair and a statically specialized ``RenderContext``
instead of looping ``mujoco.Renderer`` per environment. The physics state of
truth stays the C engine: each frame copies the derived poses the renderer
reads (``geom_xpos/geom_xmat/cam_xpos/cam_xmat/light_xpos/light_xdir``) from
the classic data bridge into the shadow warp Data, refits the scene BVH, and
launches one batched ray-tracing megakernel over all worlds.

Depth is natively available: ``rc.depth_data`` is metric planar depth matching
mjr's ``enable_depth_rendering`` output. Segmentation is not wired yet
(NotImplementedError).

A warp CPU launch is single-threaded, so the default configuration shards the
batch over forked worker processes: contexts are built and their kernels
compiled before forking, then each frame the parent ships every worker its
contiguous chunk of pose arrays and receives uint8 RGB + float32 depth back.
Worker output is bit-identical to an in-process render. Set
``MJLAB_RENDER_WORKERS`` to override the worker count (1 = in-process, no
forking); small batches always render in-process.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import threading
import time
import traceback
import warnings
import weakref
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor

# Per-frame pose fields the renderer consumes (see mujoco_warp render.py);
# everything else in the warp Data stays at its put_data-initialized values.
_POSE_SHAPES = {
  "geom_xpos": ("ngeom", 3),
  "geom_xmat": ("ngeom", 3, 3),
  "cam_xpos": ("ncam", 3),
  "cam_xmat": ("ncam", 3, 3),
  "light_xpos": ("nlight", 3),
  "light_xdir": ("nlight", 3),
}

# Batches smaller than this render in-process (no fork overhead in tests /
# tiny scenes); workers are also capped so each keeps at least this many envs.
_MIN_ENVS_PER_WORKER = 8
# A frame is ~O(100ms); a worker silent this long is considered dead.
_RECV_TIMEOUT_S = 300.0

_atexit_registered = False
_live_contexts: "weakref.WeakSet[MjwarpCameraContext]" = weakref.WeakSet()


def _register_atexit() -> None:
  global _atexit_registered

  def _close_all() -> None:
    for ctx in list(_live_contexts):
      try:
        ctx.close()
      except Exception:
        pass

  if not _atexit_registered:
    atexit.register(_close_all)
    _atexit_registered = True


def _default_workers() -> int:
  override = os.environ.get("MJLAB_RENDER_WORKERS")
  if override is not None:
    try:
      return max(1, int(override))
    except ValueError:
      pass
  # One render launch is single-threaded, so give it idle cores; leave
  # headroom for the training process (physics, torch) and this process's own
  # orchestrating thread.
  return max(1, min(4, (os.cpu_count() or 1) // 3))


def is_available() -> bool:
  """Whether the mjwarp render backend can be used on this installation."""
  try:
    import mujoco_warp  # noqa: F401

    return True
  except ImportError:
    return False


def _import_mjwarp():
  try:
    import mujoco_warp
    from mujoco_warp._src.bvh import refit_bvh

    return mujoco_warp, refit_bvh
  except ImportError as e:
    raise ImportError(
      "The mjwarp camera render backend requires mujoco_warp; install it or "
      'use CameraSensorCfg(render_backend="gl").'
    ) from e


class _WarpScene:
  """A shadow warp Model/Data + RenderContext over a contiguous env chunk."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    camera_sensors: list["CameraSensor"],
    num_envs: int,
  ) -> None:
    mujoco_warp, self._refit_bvh = _import_mjwarp()
    self._mjwarp = mujoco_warp
    self._mj_model = mj_model
    self.num_envs = num_envs

    mjm = mj_model
    self._nlight = mjm.nlight
    self._pose_fields = [f for f in _POSE_SHAPES if f != "light_xpos" or self._nlight]

    self._m = mujoco_warp.put_model(mjm)
    # Scratch data seeds the warp Data (poses are overwritten every frame).
    scratch = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, scratch)
    self._d = mujoco_warp.put_data(mjm, scratch, nworld=num_envs)

    ref_cfg = camera_sensors[0].cfg
    cam_active = [False] * mjm.ncam
    cam_res: list[tuple[int, int]] = []
    render_rgb: list[bool] = []
    render_depth: list[bool] = []
    for sensor in camera_sensors:
      cam_active[sensor.camera_idx] = True
      cam_res.append((sensor.cfg.width, sensor.cfg.height))
      render_rgb.append("rgb" in sensor.cfg.data_types)
      render_depth.append("depth" in sensor.cfg.data_types)

    # rc camera order = ascending model camera id among active cameras, which
    # equals the sorted-by-camera_idx sensor order.
    with wp.ScopedDevice("cpu"):
      self._rc = mujoco_warp.create_render_context(
        mjm=mjm,
        nworld=num_envs,
        cam_res=cam_res,
        render_rgb=render_rgb,
        render_depth=render_depth,
        render_seg=[False] * len(camera_sensors),
        use_textures=ref_cfg.use_textures,
        use_shadows=ref_cfg.use_shadows,
        enabled_geom_groups=sorted(set(ref_cfg.enabled_geom_groups)),
        cam_active=cam_active,
        render_skybox=True,
        background_color=(0.0, 0.0, 0.0, 1.0),
      )

    rgb_adr = self._rc.rgb_adr.numpy().tolist()
    depth_adr = self._rc.depth_adr.numpy().tolist()
    # Per sensor: (rgb offset, depth offset, H, W); offset -1 = not rendered.
    self._cam_slices = [
      (rgb_adr[i], depth_adr[i], s.cfg.height, s.cfg.width)
      for i, s in enumerate(camera_sensors)
    ]
    self._rgb_packed = self._rc.rgb_data.numpy()
    self._depth_flat = self._rc.depth_data.numpy()

    # Warm render: compiles the megakernel in this process. Must happen before
    # any fork so workers never touch the kernel-build machinery.
    self.render_poses(self._seed_poses(scratch))

  def _seed_poses(self, scratch: mujoco.MjData) -> dict[str, np.ndarray]:
    """Initial pose set from a forwarded scratch MjData, tiled to nworld."""
    poses: dict[str, np.ndarray] = {}
    for name in self._pose_fields:
      arr = np.asarray(getattr(scratch, name), dtype=np.float32).reshape(
        -1, *_POSE_SHAPES[name][1:]
      )
      poses[name] = np.tile(arr[None], (self.num_envs,) + (1,) * arr.ndim)
    return poses

  def render_poses(
    self, poses: dict[str, np.ndarray]
  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render one frame from per-field ``[chunk, ...]`` pose arrays.

    Returns per-sensor ``(rgb, depth)`` numpy arrays of shapes
    ``[chunk, H, W, 3]`` uint8 and ``[chunk, H, W]`` float32. Sensors whose
    cfg lacks the respective data type yield an empty list entry placement:
    the returned lists contain entries only for sensors that render that type
    (in sensor order).
    """
    for name, arr in poses.items():
      getattr(self._d, name).assign(arr)
    self._refit_bvh(self._m, self._d, self._rc)
    self._mjwarp.render(self._m, self._d, self._rc)

    rgb_out: list[np.ndarray] = []
    depth_out: list[np.ndarray] = []
    for rgb_adr, depth_adr, height, width in self._cam_slices:
      if rgb_adr >= 0:
        packed = self._rgb_packed[:, rgb_adr : rgb_adr + height * width]
        r = ((packed >> 16) & 0xFF).astype(np.uint8)
        g = ((packed >> 8) & 0xFF).astype(np.uint8)
        b = (packed & 0xFF).astype(np.uint8)
        rgb_out.append(
          np.stack([r, g, b], axis=-1).reshape(self.num_envs, height, width, 3)
        )
      if depth_adr >= 0:
        depth_out.append(
          self._depth_flat[:, depth_adr : depth_adr + height * width].reshape(
            self.num_envs, height, width
          )
        )
    return rgb_out, depth_out


def _serve_worker(scene: _WarpScene, conn) -> None:
  """Loop body for a forked render worker owning one :class:`_WarpScene`."""
  try:
    while True:
      poses = conn.recv()
      if poses is None:
        break
      rgb, depth = scene.render_poses(poses)
      conn.send(("ok", rgb, depth))
  except (EOFError, KeyboardInterrupt):
    pass
  except Exception:
    try:
      conn.send(("error", traceback.format_exc()))
    except Exception:
      pass


class MjwarpCameraContext:
  """Batched RGB+depth camera rendering through mujoco_warp (CPU).

  Same interface as :class:`mjlab.sensor.camera_cpu.CpuCameraContext`; can be
  wrapped by :class:`mjlab.sensor.camera_cpu.AsyncCpuCameraContext`, which
  drives :meth:`_render_states` from its render thread.
  """

  supports_depth = True

  def __init__(self, mj_model: mujoco.MjModel, camera_sensors) -> None:
    _import_mjwarp()  # fail fast with the backend-specific message
    self._mj_model = mj_model
    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    self._validate()

    if mj_model.nflex:
      raise NotImplementedError(
        "The mjwarp camera render backend does not bridge flex geometry yet."
      )

    # Built lazily on first render, when num_envs is known (mirrors how
    # CpuCameraContext learns num_envs from render()).
    self._scratch = mujoco.MjData(mj_model)
    self._num_envs: int | None = None
    self._rgb: list[torch.Tensor | None] = [None] * len(self.camera_sensors)
    self._depth: list[torch.Tensor | None] = [None] * len(self.camera_sensors)
    self._worker_limit: int = _default_workers()
    self._scenes: list[_WarpScene] = []
    self._chunks: list[tuple[int, int]] = []
    self._procs: list[tuple[mp.Process, Any]] = []
    self._in_band_scene: _WarpScene | None = None
    self._staging: dict[str, np.ndarray] | None = None
    self._closed = False
    self._error: BaseException | None = None
    self._build_lock = threading.Lock()

    _register_atexit()
    _live_contexts.add(self)

  # -- public API -----------------------------------------------------------

  @property
  def _pose_fields(self) -> list[str]:
    """Pose field names present for this model (lights only when nlight>0)."""
    fields = [f for f in _POSE_SHAPES if not f.startswith("light_")]
    if self._mj_model.nlight:
      fields += ["light_xpos", "light_xdir"]
    return fields

  def render(self, data, num_envs: int) -> None:
    """Render every sensor camera for all environments (main thread)."""
    self._ensure_built(num_envs)
    self._check_open()
    poses = {
      name: np.ascontiguousarray(getattr(data, name).numpy())
      for name in self._pose_fields
    }
    self._render_poses(poses)

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    sensor, buf = self._sensor_buffer(cam_idx, "rgb")
    del sensor
    assert buf is not None
    return buf

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    sensor, buf = self._sensor_buffer(cam_idx, "depth")
    del sensor
    assert buf is not None
    return buf

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    sensor, _ = self._sensor_buffer(cam_idx, "rgb")
    raise NotImplementedError(
      "Segmentation rendering is not wired for the mjwarp classic-camera "
      f"backend yet (camera '{sensor.cfg.name}')."
    )

  def close(self) -> None:
    """Shut down worker processes. Idempotent; safe from any thread."""
    with self._build_lock:
      already = self._closed
      self._closed = True
    if already:
      return
    self._shutdown_procs()
    self._scenes.clear()
    _live_contexts.discard(self)

  # -- async wrapper support -------------------------------------------------

  def _render_states(
    self,
    states: dict[str, np.ndarray],
    num_envs: int,
    out: list[torch.Tensor | None],
    depth_out: list[torch.Tensor | None] | None = None,
  ) -> None:
    """Render one frame batch from per-field ``[num_envs, ...]`` state arrays.

    Recomputes kinematics in a scratch ``MjData`` (C engine, including
    ``mj_camlight`` for static cameras/lights) and runs the same pose pipeline
    as :meth:`render`. Writes per-sensor ``[num_envs, H, W, 3]`` uint8 into
    ``out`` and ``[num_envs, H, W, 1]`` float32 depth into ``depth_out``
    (allocating buffers when None). Called on the async render thread.
    """
    self._ensure_built(num_envs)
    self._check_open()
    staging = self._staging
    assert staging is not None
    for w in range(num_envs):
      self._scratch.qpos[:] = states["qpos"][w]
      self._scratch.qvel[:] = states["qvel"][w]
      self._scratch.act[:] = states["act"][w]
      self._scratch.mocap_pos[:] = states["mocap_pos"][w]
      self._scratch.mocap_quat[:] = states["mocap_quat"][w]
      mujoco.mj_kinematics(self._mj_model, self._scratch)
      mujoco.mj_comPos(self._mj_model, self._scratch)
      mujoco.mj_camlight(self._mj_model, self._scratch)
      for name, buf in staging.items():
        arr = np.asarray(getattr(self._scratch, name), dtype=np.float32)
        buf[w] = arr.reshape(buf.shape[1:])
    self._render_poses(staging, out, depth_out)

  # -- internals ---------------------------------------------------------------

  def _validate(self) -> None:
    ref = self.camera_sensors[0].cfg
    for sensor in self.camera_sensors:
      unsupported = set(sensor.cfg.data_types) - {"rgb", "depth"}
      if unsupported:
        raise NotImplementedError(
          "The mjwarp classic-camera backend supports 'rgb' and 'depth' "
          f"(segmentation not wired yet); camera '{sensor.cfg.name}' "
          f"requested {sorted(unsupported)}."
        )
      for field in ("use_textures", "use_shadows", "enabled_geom_groups"):
        if getattr(sensor.cfg, field) != getattr(ref, field):
          raise ValueError(
            "All camera sensors must share the same "
            f"{field} ('{sensor.cfg.name}' differs from '{ref.name}')."
          )

  def _ensure_built(self, num_envs: int) -> None:
    with self._build_lock:
      if self._num_envs is not None:
        return
      if self._closed:
        raise RuntimeError(
          "The mjwarp camera context is closed; construct a new simulation "
          "to render."
        )
      workers = self._worker_limit
      if num_envs <= _MIN_ENVS_PER_WORKER:
        workers = 1
      workers = min(workers, max(1, num_envs // _MIN_ENVS_PER_WORKER))
      if workers > 1:
        self._build_sharded(num_envs, workers)
      else:
        self._build_in_band(num_envs)
      self._num_envs = num_envs

  def _build_in_band(self, num_envs: int) -> None:
    self._in_band_scene = _WarpScene(self._mj_model, self.camera_sensors, num_envs)
    self._chunks = [(0, num_envs)]
    self._staging = self._make_staging(num_envs)

  def _build_sharded(self, num_envs: int, workers: int) -> None:
    base, rem = divmod(num_envs, workers)
    self._chunks = []
    start = 0
    for i in range(workers):
      size = base + (1 if i < rem else 0)
      self._chunks.append((start, start + size))
      start += size
    try:
      # Build + compile every worker scene BEFORE forking: kernel builds must
      # never run inside a forked child.
      self._scenes = [
        _WarpScene(self._mj_model, self.camera_sensors, end - start)
        for start, end in self._chunks
      ]
      ctx = mp.get_context("fork")
      self._procs = []
      for scene in self._scenes:
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
          target=_serve_worker,
          args=(scene, child_conn),
          name="mjlab-mjwarp-render",
          daemon=True,
        )
        # The parent process is intentionally multi-threaded (physics, torch);
        # the workers only ever touch their own pre-built warp scene, so the
        # generic fork-with-threads caveat does not apply here.
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", DeprecationWarning)
          proc.start()
        child_conn.close()  # parent's copy; the child holds its own
        self._procs.append((proc, parent_conn))
      # Round-trip check: exercises the pipe and the child assign path once.
      staging = self._make_staging(num_envs)
      for (start, end), (_, conn) in zip(self._chunks, self._procs):
        conn.send({name: arr[start:end] for name, arr in staging.items()})
      for _, conn in self._procs:
        self._recv(conn)
      self._staging = staging
    except Exception as e:
      self._shutdown_procs()
      self._scenes.clear()
      self._chunks = []
      warnings.warn(
        f"mjwarp render worker pool unavailable ({e!r}); falling back to "
        "in-process rendering.",
        stacklevel=2,
      )
      self._build_in_band(num_envs)

  def _make_staging(self, num_envs: int) -> dict[str, np.ndarray]:
    """Preallocated [num_envs, ...] float32 buffers for the pose fields."""
    mjm = self._mj_model
    sizes = {"ngeom": mjm.ngeom, "ncam": mjm.ncam, "nlight": mjm.nlight}
    staging: dict[str, np.ndarray] = {}
    for name, shape in _POSE_SHAPES.items():
      if name.startswith("light_") and not mjm.nlight:
        continue
      full = tuple(sizes[dim] if isinstance(dim, str) else dim for dim in shape)
      staging[name] = np.zeros((num_envs, *full), dtype=np.float32)
    return staging

  def _render_poses(
    self,
    poses: dict[str, np.ndarray],
    out: list[torch.Tensor | None] | None = None,
    depth_out: list[torch.Tensor | None] | None = None,
  ) -> None:
    """Run one frame through the pool (or in-band) and assemble outputs.

    ``poses`` values are full-batch ``[num_envs, ...]`` arrays; each worker
    receives its contiguous chunk. ``out``/``depth_out`` default to the
    context's own sync buffers; slots that are None are allocated (fresh
    per-frame buffers, as the async wrapper hands in empty lists).
    """
    if self._procs:
      for (start, end), (_, conn) in zip(self._chunks, self._procs):
        conn.send({name: arr[start:end] for name, arr in poses.items()})
      results = [self._recv(conn) for _, conn in self._procs]
    else:
      assert self._in_band_scene is not None
      rgb, depth = self._in_band_scene.render_poses(poses)
      results = [("ok", rgb, depth)]

    out = self._rgb if out is None else out
    depth_out = self._depth if depth_out is None else depth_out
    num_envs = self._num_envs
    assert num_envs is not None
    for (start, end), result in zip(self._chunks, results):
      rgb_parts, depth_parts = result[1], result[2]
      si_rgb = si_depth = 0
      for i, sensor in enumerate(self.camera_sensors):
        h, w = sensor.cfg.height, sensor.cfg.width
        if "rgb" in sensor.cfg.data_types:
          if out[i] is None:
            out[i] = torch.zeros(num_envs, h, w, 3, dtype=torch.uint8)
          out[i][start:end] = torch.from_numpy(rgb_parts[si_rgb])
          si_rgb += 1
        if "depth" in sensor.cfg.data_types:
          if depth_out[i] is None:
            depth_out[i] = torch.zeros(num_envs, h, w, 1, dtype=torch.float32)
          depth_out[i][start:end] = torch.from_numpy(depth_parts[si_depth])[..., None]
          si_depth += 1

  def _recv(self, conn):
    deadline = time.monotonic() + _RECV_TIMEOUT_S
    while True:
      self._check_open()
      remaining = deadline - time.monotonic()
      if not conn.poll(max(0.0, remaining)):
        self._error = RuntimeError(
          f"mjwarp render worker did not answer within {_RECV_TIMEOUT_S}s."
        )
        raise self._error
      msg = conn.recv()
      if msg[0] == "error":
        self._error = RuntimeError(f"mjwarp render worker failed:\n{msg[1]}")
        raise self._error
      return msg

  def _shutdown_procs(self) -> None:
    for _, conn in self._procs:
      try:
        conn.send(None)
      except Exception:
        pass
    for proc, _ in self._procs:
      proc.join(timeout=5.0)
      if proc.is_alive():
        proc.terminate()
    self._procs.clear()

  def _sensor_buffer(self, cam_idx: int, kind: str):
    list_idx = next(
      (i for i, s in enumerate(self.camera_sensors) if s.camera_idx == cam_idx),
      None,
    )
    if list_idx is None:
      available = [s.camera_idx for s in self.camera_sensors]
      raise KeyError(
        f"Camera ID {cam_idx} not found. Available camera IDs: {available}"
      )
    sensor = self.camera_sensors[list_idx]
    if kind not in sensor.cfg.data_types:
      raise RuntimeError(
        f"Camera '{sensor.cfg.name}' does not have {kind} rendering enabled."
      )
    buf = self._rgb[list_idx] if kind == "rgb" else self._depth[list_idx]
    if buf is None:
      raise RuntimeError(
        "No mjwarp camera frame available yet; call sim.sense() before "
        "reading camera data."
      )
    return sensor, buf

  def _check_open(self) -> None:
    if self._closed:
      raise RuntimeError(
        "The mjwarp camera context is closed (sim.close() was called); "
        "construct a new simulation to render."
      )
    if self._error is not None:
      raise self._error
