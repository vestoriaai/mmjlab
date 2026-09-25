"""CPU camera rendering for the classic backend via ``mujoco.Renderer``.

One offscreen ``mujoco.Renderer`` per resolution is shared across the batch;
each environment's state is copied into a scratch ``MjData``, kinematics are
recomputed, and the scene is rendered one environment at a time. Output
semantics match the Warp path: per-sensor RGB buffers of shape
``[num_envs, height, width, 3]`` (uint8), served through ``get_rgb``.

Limitations vs the Warp render pipeline: depth and segmentation are not
available (``NotImplementedError``), ``use_textures`` is not configurable
(the model renders with its own textures), and rendering is serial per
environment — for visual RL prefer rendering every k-th step or an async
driver.
"""

from __future__ import annotations

import mujoco
import torch


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
    model = self._mj_model
    scratch = self._scratch
    for i, sensor in enumerate(self.camera_sensors):
      height, width = sensor.cfg.height, sensor.cfg.width
      buf = self._rgb[i]
      if buf is None or buf.shape[0] != num_envs:
        buf = torch.zeros(num_envs, height, width, 3, dtype=torch.uint8)
        self._rgb[i] = buf

      renderer = self._get_renderer(height, width)
      scene_option = self._scene_option(sensor)
      scene_flags = self._scene_flags(sensor)
      cam_id = sensor.camera_idx

      for w in range(num_envs):
        scratch.qpos[:] = data.qpos[w].numpy()
        scratch.qvel[:] = data.qvel[w].numpy()
        scratch.act[:] = data.act[w].numpy()
        scratch.mocap_pos[:] = data.mocap_pos[w].numpy()
        scratch.mocap_quat[:] = data.mocap_quat[w].numpy()
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        renderer.update_scene(
          scratch, camera=cam_id, scene_option=scene_option, scene_flags=scene_flags
        )
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

  def _scene_flags(self, sensor) -> int:
    flags = 0
    if sensor.cfg.use_shadows:
      flags |= int(mujoco.mjtRndFlag.mjRND_SHADOW)
    return flags

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
