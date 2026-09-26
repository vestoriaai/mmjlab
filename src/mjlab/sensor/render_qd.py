"""Batched camera rendering for the classic backend via the Quadrants (qd)
ray-tracing renderer on Metal/CUDA (stage 1, M1.5).

:class:`QdCameraContext` mirrors :class:`mjlab.sensor.render_mjwarp.
MjwarpCameraContext`'s surface exactly (``render`` / ``get_rgb`` /
``get_depth`` / ``_render_states`` / ``close`` / ``supports_depth``) so it can
be selected through ``CameraSensorCfg.render_backend="qd"`` and wrapped by the
async render thread unchanged. The rendering itself is the stage-1A/1B
validated ``qd_render_poc`` megakernel (one launch per resolution group:
ray generation + closest-hit over primitives/mesh BVH/hfield + bilinear
texture sampling + per-light Phong + any-hit shadow rays), with outputs
resident on the qd device and read back once per frame.

Alignment: vs ``mujoco.Renderer`` golden frames the renderer meets the stage-0
tolerances (RGB uint8 mean abs diff <= 2; depth background sets identical) on
all golden scenes; the residual classes (shadow-map edge vs ray-traced binary
shadow, mipmap minification vs bilinear) are shared with the mjwarp backend
and documented in ``qd-render-poc/worklog-M13.md``.

qd runtime: initialized once per process and shared with the qd raycast
backend (``MJLAB_RAYCAST_QD_ARCH`` overrides the arch selection; Metal first
on macOS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor

_POSE_SHAPES = {
  "geom_xpos": ("ngeom", 3),
  "geom_xmat": ("ngeom", 3, 3),
  "cam_xpos": ("ncam", 3),
  "cam_xmat": ("ncam", 3, 3),
  "light_xpos": ("nlight", 3),
  "light_xdir": ("nlight", 3),
}


def is_available() -> bool:
  """Whether the qd camera render backend could be used on this install."""
  try:
    import qd_render_poc  # noqa: F401
    import quadrants  # noqa: F401
  except ImportError:
    return False
  return True


class _QdCameraGroup:
  """One qd renderer over sensors sharing resolution, fovy and projection."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    sensors: list["CameraSensor"],
    num_envs: int,
  ) -> None:
    from qd_render_poc.render import QdSceneRenderer

    from mjlab.sensor import raycast_qd

    if not raycast_qd._ensure_init():
      raise RuntimeError(
        "The qd camera render backend could not initialize the quadrants "
        "runtime (see MJLAB_RAYCAST_QD_ARCH)."
      )

    ref = sensors[0]
    self.sensors = sensors
    self.num_envs = num_envs
    self._renderer = QdSceneRenderer(
      mj_model,
      nworld=num_envs,
      width=ref.cfg.width,
      height=ref.cfg.height,
      use_textures=ref.cfg.use_textures,
      use_shadows=ref.cfg.use_shadows,
      enabled_geom_groups=sorted(set(ref.cfg.enabled_geom_groups)),
      camera_id=ref.camera_idx,
      ncam=len(sensors),
    )

  def render(
    self, poses: dict[str, np.ndarray]
  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One launch for the whole group; returns per-sensor
    ``[num_envs, H, W, 3]`` uint8 and ``[num_envs, H, W]`` float32 arrays
    (sensor order)."""
    r = self._renderer
    r.set_geom_poses(poses["geom_xpos"], poses["geom_xmat"])
    if r.nlight:
      r.set_light_poses(poses["light_xpos"], poses["light_xdir"])
    # 相机位姿：第 s 个传感器取模型相机 cam 行；输出按 (S, NW) 排布
    cam_idx = np.asarray([s.camera_idx for s in self.sensors], dtype=np.int64)
    cam_xpos = poses["cam_xpos"].reshape(r.nworld, -1, 3)[:, cam_idx, :]
    cam_xmat = poses["cam_xmat"].reshape(r.nworld, -1, 9)[:, cam_idx, :]
    r.set_cam_poses(
      cam_xpos.transpose(1, 0, 2).reshape(r.S * r.nworld, 3),
      cam_xmat.transpose(1, 0, 2).reshape(r.S * r.nworld, 9),
    )
    rgb, depth = r.render()
    imgs = rgb.to_numpy().reshape(r.S, r.nworld, r.height, r.width, 3)
    deps = depth.to_numpy().reshape(r.S, r.nworld, r.height, r.width)
    return (
      [np.ascontiguousarray(imgs[s]) for s in range(r.S)],
      [np.ascontiguousarray(deps[s]) for s in range(r.S)],
    )


class QdCameraContext:
  """Batched RGB+depth camera rendering through the qd ray-tracing renderer.

  Same interface as :class:`mjlab.sensor.render_mjwarp.MjwarpCameraContext`;
  can be wrapped by :class:`mjlab.sensor.camera_cpu.AsyncCpuCameraContext`.
  """

  supports_depth = True

  def __init__(self, mj_model: mujoco.MjModel, camera_sensors) -> None:
    from mjlab.sensor import raycast_qd

    if not is_available():
      raise ImportError(
        "The qd camera render backend requires the quadrants and "
        "qd_render_poc packages; install them or use "
        'CameraSensorCfg(render_backend="mjwarp"/"gl").'
      )
    raycast_qd._ensure_init()  # fail fast on arch init failure

    self._mj_model = mj_model
    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    self._validate()

    if mj_model.nflex:
      raise NotImplementedError(
        "The qd camera render backend does not support flex geometry yet."
      )

    self._scratch = mujoco.MjData(mj_model)
    self._num_envs: int | None = None
    self._rgb: list[torch.Tensor | None] = [None] * len(self.camera_sensors)
    self._depth: list[torch.Tensor | None] = [None] * len(self.camera_sensors)
    self._groups: list[_QdCameraGroup] = []
    self._group_of: list[int] = []
    self._staging: dict[str, np.ndarray] | None = None
    self._closed = False

  # -- public API -----------------------------------------------------------

  def render(self, data, num_envs: int) -> None:
    """Render every sensor camera for all environments (main thread)."""
    self._ensure_built(num_envs)
    self._check_open()
    poses = {
      name: np.ascontiguousarray(getattr(data, name).numpy(), dtype=np.float32)
      for name in self._pose_fields
    }
    self._render_poses(poses)

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    buf = self._sensor_buffer(cam_idx, "rgb")
    return buf

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    return self._sensor_buffer(cam_idx, "depth")

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    del cam_idx
    raise NotImplementedError(
      "Segmentation rendering is not wired for the qd classic-camera backend yet."
    )

  def close(self) -> None:
    """Release renderer resources (stateless backend; buffers dropped)."""
    self._closed = True
    self._groups.clear()

  # -- async wrapper support -------------------------------------------------

  def _render_states(
    self,
    states: dict[str, np.ndarray],
    num_envs: int,
    out: list[torch.Tensor | None],
    depth_out: list[torch.Tensor | None] | None = None,
  ) -> None:
    """Render one frame batch from per-field ``[num_envs, ...]`` states.

    Recomputes kinematics in a scratch MjData (C engine, including
    ``mj_camlight``) then runs the same pipeline as :meth:`render`. Writes
    per-sensor ``[num_envs, H, W, 3]`` uint8 into ``out`` and
    ``[num_envs, H, W, 1]`` float32 into ``depth_out`` (allocating when
    None). Called on the async render thread.
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

  # -- internals -------------------------------------------------------------

  @property
  def _pose_fields(self) -> list[str]:
    fields = [f for f in _POSE_SHAPES if not f.startswith("light_")]
    if self._mj_model.nlight:
      fields += ["light_xpos", "light_xdir"]
    return fields

  def _validate(self) -> None:
    ref = self.camera_sensors[0].cfg
    for sensor in self.camera_sensors:
      unsupported = set(sensor.cfg.data_types) - {"rgb", "depth"}
      if unsupported:
        raise NotImplementedError(
          "The qd classic-camera backend supports 'rgb' and 'depth' "
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
    if self._num_envs is not None:
      return
    if self._closed:
      raise RuntimeError(
        "The qd camera context is closed; construct a new simulation to render."
      )

    # 分辨率/fovy/投影相同的传感器共享一个渲染器（一次 launch）。
    def group_key(sensor: "CameraSensor"):
      cam = sensor.camera_idx
      return (
        sensor.cfg.width,
        sensor.cfg.height,
        float(self._mj_model.cam_fovy[cam]),
        int(self._mj_model.cam_projection[cam]),
      )

    groups: dict[tuple, list["CameraSensor"]] = {}
    for sensor in self.camera_sensors:
      groups.setdefault(group_key(sensor), []).append(sensor)
    self._groups = [
      _QdCameraGroup(self._mj_model, sensors, num_envs) for sensors in groups.values()
    ]
    self._group_of = [
      next(i for i, g in enumerate(self._groups) if sensor in g.sensors)
      for sensor in self.camera_sensors
    ]
    self._staging = self._make_staging(num_envs)
    self._num_envs = num_envs

  def _make_staging(self, num_envs: int) -> dict[str, np.ndarray]:
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
    out = self._rgb if out is None else out
    depth_out = self._depth if depth_out is None else depth_out
    num_envs = self._num_envs
    assert num_envs is not None
    for group in self._groups:
      imgs, deps = group.render(poses)
      for local, sensor in enumerate(group.sensors):
        i = self.camera_sensors.index(sensor)
        h, w = sensor.cfg.height, sensor.cfg.width
        if "rgb" in sensor.cfg.data_types:
          if out[i] is None:
            out[i] = torch.zeros(num_envs, h, w, 3, dtype=torch.uint8)
          out[i][...] = torch.from_numpy(imgs[local])
        if "depth" in sensor.cfg.data_types:
          if depth_out[i] is None:
            depth_out[i] = torch.zeros(num_envs, h, w, 1, dtype=torch.float32)
          depth_out[i][...] = torch.from_numpy(deps[local])[..., None]

  def _sensor_buffer(self, cam_idx: int, kind: str) -> torch.Tensor:
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
        "No qd camera frame available yet; call sim.sense() before reading camera data."
      )
    return buf

  def _check_open(self) -> None:
    if self._closed:
      raise RuntimeError(
        "The qd camera context is closed (sim.close() was called); "
        "construct a new simulation to render."
      )
