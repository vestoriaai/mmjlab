"""CPU sensor context for the classic backend.

Drop-in counterpart of :class:`mjlab.sensor.sensor_context.SensorContext`
for ``backend='classic'``: wires raycast and camera sensors to CPU
implementations (torch height scans and ``mujoco.Renderer``) instead of the
mujoco_warp render pipeline, and exposes the same ``sense()`` entry point
used by :meth:`mjlab.sim.classic.ClassicSimulation.sense`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import raycast_cpu
from mjlab.sensor.camera_cpu import CpuCameraContext
from mjlab.sensor.raycast_cpu import CpuRaycastContext
from mjlab.sensor.raycast_sensor import GridPatternCfg

if TYPE_CHECKING:
  import mujoco

  from mjlab.sensor.camera_sensor import CameraSensor
  from mjlab.sensor.raycast_sensor import RayCastSensor


class SensorContextCPU:
  """Shared sensing resources for the classic (CPU) backend."""

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    data,
    camera_sensors: list[CameraSensor],
    raycast_sensors: list[RayCastSensor],
  ) -> None:
    self._data = data
    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    self.raycast_sensors = list(raycast_sensors)

    self._raycast_ctxs: dict[str, CpuRaycastContext] = {}
    for sensor in self.raycast_sensors:
      if not isinstance(sensor.cfg.pattern, GridPatternCfg):
        raise NotImplementedError(
          f"CPU backend supports GridPatternCfg only for raycast sensors; "
          f"sensor '{sensor.cfg.name}' uses "
          f"{type(sensor.cfg.pattern).__name__}. Use backend='warp'."
        )
      self._raycast_ctxs[sensor.cfg.name] = CpuRaycastContext(mj_model, sensor, data)

    self._camera_ctx = (
      CpuCameraContext(mj_model, self.camera_sensors) if self.camera_sensors else None
    )

    # Wire up sensors to use this context.
    for sensor in self.camera_sensors:
      sensor.set_context(self)
    for sensor in self.raycast_sensors:
      sensor.set_context(self)

  @property
  def has_cameras(self) -> bool:
    return len(self.camera_sensors) > 0

  @property
  def has_raycasts(self) -> bool:
    return len(self.raycast_sensors) > 0

  def sense(self) -> None:
    """Compute all raycast sensors and render all camera sensors."""
    for sensor in self.raycast_sensors:
      ctx = self._raycast_ctxs[sensor.cfg.name]
      _, _, origins, directions = raycast_cpu.compute_world_rays(sensor)
      distances, normals_w = ctx.height_scan(origins, directions)
      raycast_cpu.finalize(sensor, distances, normals_w, origins, directions)

    if self._camera_ctx is not None:
      self._camera_ctx.render(self._data, self._data.nworld)
      # Fresh render results: drop camera data cached by a pre-sense read
      # (mirrors SensorContext.finalize()).
      for sensor in self.camera_sensors:
        sensor._invalidate_cache()

  # Camera data access, mirroring SensorContext's interface.

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    assert self._camera_ctx is not None
    return self._camera_ctx.get_rgb(cam_idx)

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    assert self._camera_ctx is not None
    return self._camera_ctx.get_depth(cam_idx)

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    assert self._camera_ctx is not None
    return self._camera_ctx.get_segmentation(cam_idx)
