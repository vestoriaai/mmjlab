"""CPU camera rendering for the classic backend via ``mujoco.Renderer``.

One offscreen ``mujoco.Renderer`` per resolution is shared across the batch;
each environment's state is copied into a scratch ``MjData``, kinematics are
recomputed, and the scene is rendered one environment at a time. Output
semantics match the Warp path: per-sensor RGB buffers of shape
``[num_envs, height, width, 3]`` (uint8), served through ``get_rgb``.

Limitations vs the Warp render pipeline: depth and segmentation are not
available (``NotImplementedError``), ``use_textures``/``use_shadows`` are
not configurable (the model renders with its own textures and the
renderer's default flags), and rendering is serial per environment — for
visual RL prefer rendering every k-th step or set
``CameraSensorCfg.async_render = True`` (classic backend), which moves
rendering to a background thread (see :class:`AsyncCpuCameraContext`).
"""

from __future__ import annotations

import atexit
import threading
import time
import weakref

import mujoco
import numpy as np
import torch

# Per-environment state fields a render needs; the async renderer snapshots
# exactly these from the batched data bridge on the main thread.
_STATE_FIELDS = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat")


class CpuCameraContext:
  """Batched RGB rendering on top of a single-process ``mujoco.Renderer``."""

  def __init__(self, mj_model: mujoco.MjModel, camera_sensors) -> None:
    self._mj_model = mj_model
    self.camera_sensors = list(camera_sensors)
    self._scratch = mujoco.MjData(mj_model)
    self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    # Per-sensor output buffers, allocated lazily on first render (num_envs
    # is known from the data bridge passed to render()).
    self._rgb: list[torch.Tensor | None] = [None] * len(self.camera_sensors)

    for sensor in self.camera_sensors:
      unsupported = set(sensor.cfg.data_types) - {"rgb"}
      if unsupported:
        raise NotImplementedError(
          "The classic backend camera sensor supports 'rgb' only; sensor "
          f"'{sensor.cfg.name}' requested {sorted(unsupported)}. Use "
          "backend='warp' for depth/segmentation."
        )

  def _get_renderer(self, height: int, width: int) -> mujoco.Renderer:
    key = (height, width)
    renderer = self._renderers.get(key)
    if renderer is None:
      renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
      self._renderers[key] = renderer
    return renderer

  def render(self, data, num_envs: int) -> None:
    """Render every sensor camera for all environments."""
    states = {name: getattr(data, name).numpy() for name in _STATE_FIELDS}
    self._render_states(states, num_envs, self._rgb)

  def _render_states(
    self,
    states: dict[str, np.ndarray],
    num_envs: int,
    out: list[torch.Tensor | None],
    depth_out: list[torch.Tensor | None] | None = None,
  ) -> None:
    """Render one frame batch from per-field ``[num_envs, ...]`` state arrays.

    Writes one ``[num_envs, H, W, 3]`` uint8 buffer per sensor into ``out``
    (allocated on shape mismatch). Called on the main thread by :meth:`render`
    and on the render thread by :class:`AsyncCpuCameraContext`; the mujoco
    resources used here (scratch ``MjData``, lazily created ``Renderer``s) are
    touched by one thread at a time only. ``depth_out`` exists for interface
    parity with render backends that support depth; this GL backend does not
    and ignores it.
    """
    del depth_out
    model = self._mj_model
    scratch = self._scratch
    for i, sensor in enumerate(self.camera_sensors):
      height, width = sensor.cfg.height, sensor.cfg.width
      buf = out[i]
      if buf is None or buf.shape[0] != num_envs:
        buf = torch.zeros(num_envs, height, width, 3, dtype=torch.uint8)
        out[i] = buf

      renderer = self._get_renderer(height, width)
      scene_option = self._scene_option(sensor)
      cam_id = sensor.camera_idx

      for w in range(num_envs):
        scratch.qpos[:] = states["qpos"][w]
        scratch.qvel[:] = states["qvel"][w]
        scratch.act[:] = states["act"][w]
        scratch.mocap_pos[:] = states["mocap_pos"][w]
        scratch.mocap_quat[:] = states["mocap_quat"][w]
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        # Static (worldbody) cameras/lights only get their mjData positions
        # from mj_camlight, not mj_kinematics, in mujoco 3.11.
        mujoco.mj_camlight(model, scratch)
        renderer.update_scene(scratch, camera=cam_id, scene_option=scene_option)
        buf[w] = torch.from_numpy(renderer.render())

  def _scene_option(self, sensor) -> mujoco.MjvOption | None:
    """Geom-group visibility matching the sensor cfg (None: MuJoCo default)."""
    groups = sensor.cfg.enabled_geom_groups
    if groups is None:
      return None
    opt = mujoco.MjvOption()
    for g in range(mujoco.mjNGROUP):
      opt.geomgroup[g] = 1 if g in set(groups) else 0
    return opt

  # Data access, mirroring SensorContext.get_rgb for CameraSensor.

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    sensor, buf = self._sensor_buffer(cam_idx)
    del sensor
    assert buf is not None
    return buf

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    sensor, _ = self._sensor_buffer(cam_idx)
    raise NotImplementedError(
      f"Depth rendering is not supported by the classic backend (camera "
      f"'{sensor.cfg.name}'); use backend='warp'."
    )

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    sensor, _ = self._sensor_buffer(cam_idx)
    raise NotImplementedError(
      f"Segmentation rendering is not supported by the classic backend "
      f"(camera '{sensor.cfg.name}'); use backend='warp'."
    )

  def _sensor_buffer(self, cam_idx: int):
    list_idx = next(
      (i for i, s in enumerate(self.camera_sensors) if s.camera_idx == cam_idx),
      None,
    )
    if list_idx is None:
      available = [s.camera_idx for s in self.camera_sensors]
      raise KeyError(
        f"Camera ID {cam_idx} not found. Available camera IDs: {available}"
      )
    return self.camera_sensors[list_idx], self._rgb[list_idx]


# Async rendering (CameraSensorCfg.async_render=True, classic backend).

# Live async contexts, for the atexit backstop below: if the owning
# ClassicSimulation is never closed (leaked reference cycle, interactive
# session end), the render thread's GL resources are still released.
_LIVE_ASYNC_CONTEXTS: "weakref.WeakSet[AsyncCpuCameraContext]" = weakref.WeakSet()
_ATTEXIT_REGISTERED = False


def _close_live_contexts_at_exit() -> None:
  for ctx in list(_LIVE_ASYNC_CONTEXTS):
    try:
      ctx.close()
    except Exception:
      pass


class AsyncCpuCameraContext:
  """Camera rendering on a background thread, wrapping ``CpuCameraContext``.

  Main thread: :meth:`render` copies the batched state into numpy snapshots,
  hands the newest snapshot to the render thread (monotonically increasing
  frame ids; stale snapshots are skipped, so the thread always renders the
  most recent state) and returns the most recently *completed* frame buffer.
  Render thread: owns every GL resource it touches — ``mujoco.Renderer``
  objects are created lazily inside the thread and closed there, since
  OpenGL contexts are thread-affine (macOS mujoco defaults to the CGL
  backend, which permits thread-local offscreen contexts). While no frame
  has completed yet, the first :meth:`render` blocks for that one frame so
  ``get_rgb`` always has data afterwards.

  Frame semantics (v1, fixed): at most one task in flight and completed
  frames are served immediately, so with a render thread that keeps up, a
  frame lags its ``sense()`` call by at most one step. If rendering is
  persistently slower than ``sense()`` is called, the returned frame can
  trail further — the main thread is never blocked (except for the first
  frame).

  Unlike the sync context, each completed frame is a fresh buffer:
  tensors returned by :meth:`get_rgb` are not overwritten in place by
  later renders.
  """

  _FIRST_FRAME_TIMEOUT_S = 30.0
  _JOIN_TIMEOUT_S = 10.0

  def __init__(self, mj_model: mujoco.MjModel, camera_sensors, inner=None) -> None:
    # Rendering internals of the wrapped context (scratch MjData, lazily
    # created Renderers / shadow warp scenes) are owned by the render thread
    # from here on. `inner` injects an alternative sync context (e.g. the
    # mjwarp render backend); default is the GL context.
    self._inner = inner if inner is not None else CpuCameraContext(mj_model, camera_sensors)
    self.camera_sensors = self._inner.camera_sensors

    self._frame_id = 0
    self._task: tuple[int, int, dict[str, np.ndarray]] | None = None
    self._latest: tuple[int, list[torch.Tensor], list[torch.Tensor] | None] | None = None
    self._stop = False
    self._closed = False
    self._thread_error: BaseException | None = None
    self._task_cond = threading.Condition()
    self._done_cond = threading.Condition()
    self._thread = threading.Thread(
      target=self._render_loop,
      name="mjlab-classic-camera-render",
      daemon=True,
    )
    self._thread.start()

    global _ATTEXIT_REGISTERED
    _LIVE_ASYNC_CONTEXTS.add(self)
    if not _ATTEXIT_REGISTERED:
      atexit.register(_close_live_contexts_at_exit)
      _ATTEXIT_REGISTERED = True

  # Main-thread API, mirroring CpuCameraContext's surface.

  def render(self, data, num_envs: int) -> int:
    """Snapshot state, submit a render task, return the completed frame.

    Returns the submitted frame id. Blocks only until the first frame has
    completed (no completed frame yet), so reads always have data.
    """
    self._check_open()
    states = {
      name: getattr(data, name).detach().numpy().copy() for name in _STATE_FIELDS
    }
    with self._task_cond:
      self._frame_id += 1
      frame_id = self._frame_id
      self._task = (frame_id, num_envs, states)  # newest wins
      self._task_cond.notify_all()
    if self._latest is None:
      self._wait_first_frame()
    return frame_id

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    self._check_open()
    latest = self._latest
    if latest is None:
      raise RuntimeError(
        "No async camera frame available yet; call sim.sense() before "
        "reading camera data."
      )
    return latest[1][self._sensor_index(cam_idx)]

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    if not getattr(self._inner, "supports_depth", False):
      raise NotImplementedError(
        f"Depth rendering is not supported by the classic backend (camera "
        f"'{self._sensor(cam_idx).cfg.name}'); use backend='warp'."
      )
    self._check_open()
    latest = self._latest
    if latest is None:
      raise RuntimeError(
        "No async camera frame available yet; call sim.sense() before "
        "reading camera data."
      )
    return latest[2][self._sensor_index(cam_idx)]

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    raise NotImplementedError(
      f"Segmentation rendering is not supported by the classic backend "
      f"(camera '{self._sensor(cam_idx).cfg.name}'); use backend='warp'."
    )

  def close(self) -> None:
    """Stop the render thread and release its renderers. Idempotent.

    The renderers are closed on the render thread itself (GL context
    affinity); if the thread is mid-frame, ``close`` joins it for up to
    ``_JOIN_TIMEOUT_S`` and the thread finishes releasing resources on exit.
    """
    with self._task_cond:
      if self._closed:
        return
      self._closed = True
      self._stop = True
      self._task_cond.notify_all()
    if self._thread is not threading.current_thread() and self._thread.is_alive():
      self._thread.join(self._JOIN_TIMEOUT_S)
    _LIVE_ASYNC_CONTEXTS.discard(self)

  # Introspection / synchronization helpers (used by tests and callers that
  # want to wait for a specific frame).

  @property
  def latest_submitted_frame_id(self) -> int:
    return self._frame_id

  @property
  def latest_completed_frame_id(self) -> int | None:
    latest = self._latest
    return None if latest is None else latest[0]

  def wait_for_frame(self, frame_id: int, timeout: float = 30.0):
    """Block until ``frame_id`` (or newer) completed; returns that frame."""
    deadline = time.monotonic() + timeout
    with self._done_cond:
      while self._latest is None or self._latest[0] < frame_id:
        if self._thread_error is not None:
          raise RuntimeError(
            "The async camera render thread failed."
          ) from self._thread_error
        if self._closed:
          raise RuntimeError(
            "The async camera context was closed while waiting for a frame."
          )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          raise TimeoutError(
            f"Async camera frame {frame_id} did not complete within {timeout}s."
          )
        self._done_cond.wait(remaining)
      return self._latest

  # Internals.

  def _check_open(self) -> None:
    if self._closed:
      raise RuntimeError(
        "The async camera context is closed (sim.close() was called or the "
        "simulation was torn down); construct a new simulation to render."
      )
    if self._thread_error is not None:
      raise RuntimeError(
        "The async camera render thread failed; see chained exception."
      ) from self._thread_error

  def _wait_first_frame(self) -> None:
    deadline = time.monotonic() + self._FIRST_FRAME_TIMEOUT_S
    with self._done_cond:
      while self._latest is None:
        if self._thread_error is not None:
          raise RuntimeError(
            "The async camera render thread failed on its first frame."
          ) from self._thread_error
        if self._closed and not self._thread.is_alive():
          raise RuntimeError(
            "The async camera context was closed before the first frame "
            "completed."
          )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          raise RuntimeError(
            "The async camera render thread produced no frame within "
            f"{self._FIRST_FRAME_TIMEOUT_S}s."
          )
        self._done_cond.wait(remaining)

  def _render_loop(self) -> None:
    try:
      while True:
        with self._task_cond:
          while self._task is None and not self._stop:
            self._task_cond.wait(timeout=0.1)
          if self._stop:
            break
          frame_id, num_envs, states = self._task
          self._task = None
        outputs: list[torch.Tensor | None] = [None] * len(self.camera_sensors)
        depth_outputs = (
          [None] * len(self.camera_sensors)
          if getattr(self._inner, "supports_depth", False)
          else None
        )
        self._inner._render_states(states, num_envs, outputs, depth_outputs)
        with self._done_cond:
          self._latest = (frame_id, outputs, depth_outputs)
          self._done_cond.notify_all()
    except BaseException as exc:  # surfaced to sense()/get_rgb callers
      with self._done_cond:
        self._thread_error = exc
        self._done_cond.notify_all()
    finally:
      # Wake first-frame / wait_for_frame waiters on any exit path (including
      # a close() that raced the very first frame).
      with self._done_cond:
        self._done_cond.notify_all()
      self._close_renderers()

  def _close_renderers(self) -> None:
    # GL renderers are thread-affine and live on self._inner._renderers;
    # other inner contexts (mjwarp) shut down through their own close().
    for renderer in getattr(self._inner, "_renderers", {}).values():
      try:
        renderer.close()
      except Exception:
        pass
    if hasattr(self._inner, "_renderers"):
      self._inner._renderers.clear()
    close = getattr(self._inner, "close", None)
    if close is not None:
      try:
        close()
      except Exception:
        pass

  def _sensor(self, cam_idx: int):
    for sensor in self.camera_sensors:
      if sensor.camera_idx == cam_idx:
        return sensor
    raise KeyError(
      f"Camera ID {cam_idx} not found. Available camera IDs: "
      f"{[s.camera_idx for s in self.camera_sensors]}"
    )

  def _sensor_index(self, cam_idx: int) -> int:
    return self.camera_sensors.index(self._sensor(cam_idx))
