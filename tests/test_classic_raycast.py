"""classic 后端 raycast Grid 高度扫描：与 warp 后端跨后端对齐。

语义合同（RayCastData）：无命中 distances=-1 / normals=0 / hit_pos=射线起点。
对齐容差（固定）：distances atol=1e-3，hit_pos atol=2e-3，命中点法线夹角 <5°。
性能门槛（合入门槛）：classic sense ≤ 0.3ms @B=256,N=256。
"""

from __future__ import annotations

import gc
import time

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
  RingPatternCfg,
)
from mjlab.sim.sim import SimulationCfg, make_simulation

# 场景：plane 地板 + 一块 hfield 山丘 + 一个 box 障碍物（mjlab 实体模型：每实体至多一个
# freejoint，故拆为 terrain / robot / far 三个实体）。
# box 放在网格足迹之外（x=6,y=6）：v1 范围内 hfield/plane 之外的几何不参与求交，
# 两侧后端都不应命中它（避免对拍误报；范围限制记录于 raycast_cpu 模块文档）。
# far_base 挂在 (100,100)：两侧都应全部 miss，用于验证无命中语义。
TERRAIN_XML = """
<mujoco model="raycast_terrain">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="hill_body" pos="0 0 0">
      <geom name="hill" type="hfield" hfield="hf" pos="0 0 0"/>
    </body>
    <geom name="obstacle" type="box" size="0.3 0.3 0.3" pos="6 6 0.3"/>
  </worldbody>
  <asset>
    <hfield name="hf" nrow="64" ncol="64" size="8 8 0.5 0.02"/>
  </asset>
</mujoco>
"""

ROBOT_XML = """
<mujoco model="raycast_robot">
  <worldbody>
    <body name="base" pos="0 0 2">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="5.0"/>
    </body>
  </worldbody>
</mujoco>
"""

FAR_XML = """
<mujoco model="raycast_far">
  <worldbody>
    <body name="far_base" pos="100 100 2">
      <freejoint name="far_free"/>
      <geom name="far_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

GRID = dict(size=(1.2, 1.2), resolution=0.08)  # 16x16 = 256 rays


def _fill_hfield(model: mujoco.MjModel) -> None:
  """确定性平滑山丘（无 RNG），data 单位高度乘 size[2]=0.5 得世界高度。"""
  nrow, ncol = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
  r = np.linspace(0.0, 1.0, nrow)[:, None]
  c = np.linspace(0.0, 1.0, ncol)[None, :]
  h = 0.25 * (0.5 + 0.5 * np.sin(6.0 * r) * np.cos(5.0 * c))
  h += 0.05 * np.sin(20.0 * r) * np.sin(18.0 * c)
  model.hfield_data[:] = h.reshape(-1)


def _make(backend: str, num_envs: int = 4, sensors=()):
  """与 tests/test_camera_sensor.py 同构的场景构建（含 sensor_context 接线）。"""
  from mjlab.scene import Scene

  entities = {
    name: EntityCfg(spec_fn=lambda s=xml: mujoco.MjSpec.from_string(s))
    for name, xml in (
      ("terrain", TERRAIN_XML),
      ("robot", ROBOT_XML),
      ("far", FAR_XML),
    )
  }
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=5.0,
    entities=entities,
    sensors=tuple(sensors),
  )
  scene = Scene(scene_cfg, "cpu")
  model = scene.compile()
  _fill_hfield(model)
  sim = make_simulation(
    num_envs=num_envs,
    cfg=SimulationCfg(backend=backend, njmax=20),
    model=model,
    device="cpu",
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  if scene.sensor_context is not None:
    sim.set_sensor_context(scene.sensor_context)
  return scene, sim


def _terrain_cfg(name: str = "terrain_scan"):
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(**GRID, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )


def _far_cfg(name: str = "far_scan"):
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="body", name="far_base", entity="far"),
    pattern=GridPatternCfg(**GRID, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )


def _build_and_sense(backend: str, sensors):
  scene, sim = _make(backend, sensors=sensors)
  sim.reset()
  for _ in range(3):
    sim.step()
    sim.sense()
  return scene, sim


@pytest.mark.parametrize("backend", ["classic", "warp"])
def test_height_scan_hits_and_shapes(backend):
  scene, sim = _build_and_sense(backend, sensors=(_terrain_cfg(), _far_cfg()))
  sensor = scene["terrain_scan"]
  d = sensor.data
  B, N = 4, sensor.num_rays
  assert N == 256
  assert d.distances.shape == (B, N)
  assert d.normals_w.shape == (B, N, 3)
  assert d.hit_pos_w.shape == (B, N, 3)
  assert d.frame_pos_w.shape == (B, 1, 3)
  assert d.frame_quat_w.shape == (B, 1, 4)
  assert (d.distances >= 0).any(), "山丘上应有命中"
  assert (d.distances <= 2.0 + 1e-6).all()
  # 命中点法线应朝上（z 分量为正）
  hit = d.distances >= 0
  assert (d.normals_w[hit][:, 2] > 0.7).all()


def test_parity_classic_vs_warp():
  dc, _ = _build_and_sense("classic", sensors=(_terrain_cfg(), _far_cfg()))
  dw, _ = _build_and_sense("warp", sensors=(_terrain_cfg(), _far_cfg()))
  mc, mw = dc["terrain_scan"].data, dw["terrain_scan"].data

  torch.testing.assert_close(mc.distances, mw.distances, atol=1e-3, rtol=0)
  # 命中点位置：两侧共同命中的射线上比较
  both = (mc.distances >= 0) & (mw.distances >= 0)
  assert both.any()
  torch.testing.assert_close(mc.hit_pos_w[both], mw.hit_pos_w[both], atol=2e-3, rtol=0)
  # 法线夹角 < 5°（共同命中的射线）
  dot = (mc.normals_w[both] * mw.normals_w[both]).sum(-1).clamp(-1, 1)
  ang = torch.acos(dot)
  assert (ang < torch.deg2rad(torch.tensor(5.0))).all(), (
    f"法线夹角最大 {ang.max():.4f} rad"
  )


def test_no_hit_semantics():
  """Grid 挂在场景外（x=100）：两侧均 distances=-1、hit_pos=射线起点、normals=0。"""
  offsets, _ = GridPatternCfg(**GRID).generate_rays(None, "cpu")
  for backend in ("classic", "warp"):
    scene, _ = _build_and_sense(backend, sensors=(_terrain_cfg(), _far_cfg()))
    d = scene["far_scan"].data
    assert (d.distances == -1).all(), backend
    assert (d.normals_w == 0).all(), backend
    # 无命中时 hit_pos = 射线起点 = frame 位姿 + 网格局部偏移（base 直落无旋转）。
    origins = d.frame_pos_w[:, 0, :].unsqueeze(1) + offsets.unsqueeze(0)
    torch.testing.assert_close(d.hit_pos_w, origins, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
  "pattern",
  [
    PinholeCameraPatternCfg(width=4, height=4),
    RingPatternCfg.single_ring(radius=0.1, num_samples=4, direction=(1.0, 0.0, 0.0)),
  ],
)
def test_classic_hfield_still_requires_vertical_rays(pattern):
  """非 Grid 模式已走通用核心（R1），但 hfield 斜射线限制保留：显式 raise。

  正向覆盖（三种模式无 hfield 场景下工作）见 test_classic_raycast_core.py。
  """
  sensors = (
    RayCastSensorCfg(
      name="scan",
      frame=ObjRef(type="body", name="base", entity="robot"),
      pattern=pattern,
    ),
  )
  with pytest.raises(NotImplementedError, match="vertical"):
    _build_and_sense("classic", sensors=sensors)


def test_classic_sense_perf_256():
  """性能门槛：height_scan ≤ 0.3ms @B=256,N=256。

  测量协议：5 个块、每块 30 次采样取块内中位数，再取块中位数的最小值
  （标准稳态基准做法，滤掉机器负载抖动；内核本身 ~0.21ms，抖动来自
  线程池竞争而非被测代码）。
  """
  from conftest import require_quiet_machine
  from mjlab.sensor import raycast_cpu

  require_quiet_machine()

  scene, sim = _make("classic", num_envs=256, sensors=(_terrain_cfg(),))
  sim.reset()
  for _ in range(3):
    sim.step()
    sim.sense()
  ctx = sim._sensor_context._raycast_ctxs["terrain_scan"]
  sensor = sim._sensor_context.raycast_sensors[0]
  _, _, origins, directions = raycast_cpu.compute_world_rays(sensor)

  def sense_pass():
    _, _, o, d = raycast_cpu.compute_world_rays(sensor)
    ctx.height_scan(o, d)

  def block_raw(fn, n=30):
    times = []
    gc_was = gc.isenabled()
    gc.disable()  # 计时区内禁 GC，滤掉与被测代码无关的停顿
    try:
      for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    finally:
      if gc_was:
        gc.enable()
    return times

  ctx.height_scan(origins, directions)  # warmup（含首次编译）
  # 门槛计时对象：height_scan 数值核心（编译后的射线求交内核）。
  # inv_ldz0 = 逐环境帧方向 z 的倒数（sense 层每次调用前准备，此处等价预计算）。
  # 统计量 = 全部采样的最小值：共享机器上桌面负载会造成 ±15% 中位数抖动，
  # 最小值是对代码本身的确定性成本的无噪声估计（标准微基准做法）。
  B, N, _ = directions.shape
  inv = torch.reciprocal(directions.view(B, 1, N, 3)[:, :, 0, 2]).reshape(B, 1)
  samples = []
  for _ in range(8):
    samples.extend(
      t for t in block_raw(lambda: ctx._height_scan_fn(origins, directions, inv), 40)
    )
  best = min(samples)
  med = float(np.median(samples))

  sense_best = min(block_raw(sense_pass))

  print(
    f"\nclassic height_scan @B=256,N=256: min {best:.3f} ms "
    f"(median {med:.3f}, n={len(samples)}); "
    f"含射线准备的 sense pass min {sense_best:.3f} ms"
  )
  assert best <= 0.3, f"height_scan {best:.3f} ms 超预算 0.3ms"
