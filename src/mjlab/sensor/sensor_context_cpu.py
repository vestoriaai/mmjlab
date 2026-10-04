"""CPU sensor context for the classic backend.

Drop-in counterpart of :class:`mjlab.sensor.sensor_context.SensorContext`
for ``backend='classic'``: wires raycast and camera sensors to CPU
implementations (the general torch ray-intersection core in
:mod:`mjlab.sensor.raycast_core` and ``mujoco.Renderer``) instead of the
mujoco_warp render pipeline, and exposes the same ``sense()`` entry point
used by :meth:`mjlab.sim.classic.ClassicSimulation.sense`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import raycast_cpu, raycast_qd
from mjlab.sensor.camera_cpu import AsyncCpuCameraContext, CpuCameraContext
from mjlab.sensor.raycast_core import RaycastCoreContext

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

    self._raycast_ctxs: dict[str, RaycastCoreContext | raycast_qd.QdRaycastContext] = {}
    for sensor in self.raycast_sensors:
      self._raycast_ctxs[sensor.cfg.name] = raycast_qd.try_build_context(
        mj_model, sensor, data
      ) or RaycastCoreContext(mj_model, sensor, data)

    # Async camera rendering (CameraSensorCfg.async_render) is opt-in per
    # sensor; if any sensor opts in, all cameras render on the background
    # thread (each frame renders every sensor anyway).
    self._camera_ctx = None
    if self.camera_sensors:
      self._camera_ctx = self._build_camera_context(mj_model)

    # Wire up sensors to use this context.
    for sensor in self.camera_sensors:
      sensor.set_context(self)
    for sensor in self.raycast_sensors:
      sensor.set_context(self)

  def _build_camera_context(self, mj_model: mujoco.MjModel):
    """Select the sync camera context by ``render_backend`` and wrap it in
    the async render thread when any sensor opts in.

    ``"auto"`` prefers the qd (Quadrants) batch renderer when importable,
    then the mjwarp batch renderer, falling back to GL; ``"qd"`` /
    ``"mjwarp"`` / ``"gl"`` force one path.
    """
    from mjlab.sensor.render_mjwarp import MjwarpCameraContext, is_available
    from mjlab.sensor.render_qd import QdCameraContext
    from mjlab.sensor.render_qd import is_available as qd_available

    backends = {s.cfg.render_backend for s in self.camera_sensors}
    if len(backends) > 1:
      raise ValueError(
        "All camera sensors must share the same render_backend; got "
        f"{sorted(backends)}."
      )
    backend = backends.pop()
    if backend == "auto":
      backend = "qd" if qd_available() else ("mjwarp" if is_available() else "gl")
    if backend == "qd":
      # MuJoCo 默认天空渐变（无 skybox 时的内建蓝色渐变，与 mjr/mjwarp 对齐；
      # 采样自 mjwarp 实测剖面：顶部 (54,108,160) → 地平线 (116,171,228)）
      inner = QdCameraContext(
        mj_model, self.camera_sensors,
        sky_gradient=((54 / 255, 108 / 255, 160 / 255), (116 / 255, 171 / 255, 228 / 255)),
      )
    elif backend == "mjwarp":
      inner = MjwarpCameraContext(mj_model, self.camera_sensors)
    else:
      inner = CpuCameraContext(mj_model, self.camera_sensors)
    if any(s.cfg.async_render for s in self.camera_sensors):
      return AsyncCpuCameraContext(mj_model, self.camera_sensors, inner=inner)
    return inner

  @property
  def has_cameras(self) -> bool:
    return len(self.camera_sensors) > 0

  @property
  def has_raycasts(self) -> bool:
    return len(self.raycast_sensors) > 0

  @property
  def camera_context(self):
    """The camera render context (``CpuCameraContext`` or async wrapper)."""
    return self._camera_ctx

  def close(self) -> None:
    """Release background render resources (async camera render thread)."""
    if self._camera_ctx is not None and hasattr(self._camera_ctx, "close"):
      self._camera_ctx.close()

  def sense(self) -> None:
    """Compute all raycast sensors and render all camera sensors."""
    for sensor in self.raycast_sensors:
      ctx = self._raycast_ctxs[sensor.cfg.name]
      if isinstance(ctx, raycast_qd.QdRaycastContext):
        ctx.sense(sensor)  # qd path: fused raygen + intersect + finalize
        continue
      _, _, origins, directions = raycast_cpu.compute_world_rays(sensor)
      distances, normals_w, _ = ctx.closest_hit(origins, directions)
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
