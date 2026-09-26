"""classic 后端通用射线求交核心（raycast_core）：mesh 命中 × Grid/Pinhole/Ring 三模式。

对齐合同（固定，不许放宽）：
- mesh 命中 vs ``mujoco.mj_ray`` 逐射线 f64 参考（scratch MjData + mj_kinematics）：
  命中/无命中模式完全一致，命中距离 rel < 1e-4。
- 跨后端 classic vs warp（沿用 F-track 口径，场景含 mesh geom 在场）：
  distances atol=1e-3，共同命中 hit_pos atol=2e-3，法线夹角 < 5°。
- 无命中语义（RayCastData）：distances=-1 / normals=0 / hit_pos=射线起点。
性能：路线图 R1 门槛 sense ≤ 1ms @256 envs × 720 rays + 2028 三角 mesh **未达**
（torch CPU 逐 op 分发受限，实测 min ≈ 16ms，优化过程见 docs/results/
worklog-R1.md）——按路线图预案标注「建议走阶段 1 GPU 射线核心」；本文件的
性能测试为回归线（防数量级劣化），非门槛放宽。

mesh 求交为单面（背面剔除），与 warp sensor 路径（ray_mesh_with_bvh 的
cull_backfaces=True）一致；C 端 mj_ray 是双面的——本文件所有 mj_ray 参考射线
均从实体外部发射，两者在首个正面命中上等价。
"""

from __future__ import annotations

import dataclasses
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

# ---------------------------------------------------------------------------
# Mesh helpers: MjSpec.add_mesh + uservert/userface（OBJ 文件资产经 Scene 合并
# 会丢失 assets，改用内联顶点/面）。
# ---------------------------------------------------------------------------


def _box_mesh_arrays(half: float, n: int) -> tuple[np.ndarray, np.ndarray]:
  """闭合轴对齐盒表面，每面 n×n 四边形网格（2 三角/格）= 12*n^2 三角形。

  外向绕序（cross(du,dv) 指向面外侧），顶点在原点居中。
  """
  verts: list[np.ndarray] = []
  faces: list[np.ndarray] = []
  h = half
  # (origin, du, dv)：cross(du, dv) = 面外法线，origin 在对应面上。
  face_axes = [
    ((h, -h, -h), (0, 2 * h, 0), (0, 0, 2 * h)),  # +x
    ((-h, h, -h), (0, -2 * h, 0), (0, 0, 2 * h)),  # -x
    ((-h, h, -h), (0, 0, 2 * h), (2 * h, 0, 0)),  # +y
    ((h, -h, -h), (0, 0, 2 * h), (-2 * h, 0, 0)),  # -y
    ((-h, -h, h), (2 * h, 0, 0), (0, 2 * h, 0)),  # +z
    ((-h, -h, -h), (0, 2 * h, 0), (2 * h, 0, 0)),  # -z
  ]
  for origin, du, dv in face_axes:
    origin = np.asarray(origin, dtype=np.float64)
    du = np.asarray(du, dtype=np.float64) / n
    dv = np.asarray(dv, dtype=np.float64) / n
    base = len(verts)
    for i in range(n + 1):
      for j in range(n + 1):
        verts.append(origin + du * i + dv * j)
    for i in range(n):
      for j in range(n):
        a = base + i * (n + 1) + j
        b = base + (i + 1) * (n + 1) + j
        c = base + (i + 1) * (n + 1) + j + 1
        d = base + i * (n + 1) + j + 1
        faces.append((a, b, c))
        faces.append((a, c, d))
  return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _mesh_entity(
  name: str,
  body: str,
  geom: str,
  pos: str,
  quat: str | None,
  verts: np.ndarray,
  faces: np.ndarray,
  mass: float = 2.0,
):
  """带 mesh geom 的 mjlab 实体（freejoint 动体；pos/quat 为初始位姿）。"""
  quat_attr = f' quat="{quat}"' if quat else ""
  xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{body}" pos="{pos}"{quat_attr}>
      <freejoint name="{body}_free"/>
      <geom name="{geom}" type="mesh" mesh="{name}_mesh" mass="{mass}"/>
    </body>
  </worldbody>
</mujoco>
"""

  def spec_fn():
    spec = mujoco.MjSpec.from_string(xml)
    mesh = spec.add_mesh()
    mesh.name = f"{name}_mesh"
    mesh.uservert = np.asarray(verts, dtype=np.float32).reshape(-1)
    mesh.userface = np.asarray(faces, dtype=np.int32).reshape(-1)
    return spec

  return EntityCfg(spec_fn=spec_fn)


def _static_mesh_entity(
  name: str, body: str, geom: str, pos: str, verts: np.ndarray, faces: np.ndarray
):
  """无关节静态 body 上的 mesh geom（位姿分类 constant 模式）。"""
  xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{body}" pos="{pos}">
      <geom name="{geom}" type="mesh" mesh="{name}_mesh"/>
    </body>
  </worldbody>
</mujoco>
"""

  def spec_fn():
    spec = mujoco.MjSpec.from_string(xml)
    mesh = spec.add_mesh()
    mesh.name = f"{name}_mesh"
    mesh.uservert = np.asarray(verts, dtype=np.float32).reshape(-1)
    mesh.userface = np.asarray(faces, dtype=np.int32).reshape(-1)
    return spec

  return EntityCfg(spec_fn=spec_fn)


# ---------------------------------------------------------------------------
# 场景。
# ---------------------------------------------------------------------------

CRATE_HALF = 0.4
CRATE_QUAT = "0.9239 0 0 0.3827"  # 绕 z 45°，动体旋转 BVH 路径
PILLAR_HALF = 0.15

# base z=1.0；cam_site 前向 +x 下俯 20°（Pinhole 对准 crate），up_site 朝天。
ROBOT_XML = """
<mujoco model="raycast_robot">
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="5.0"/>
      <site name="cam_site" pos="0 0 0" quat="@CAM_QUAT@"/>
      <site name="up_site" pos="0 0 0" quat="0 1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _cam_quat() -> str:
  """cam_site 四元数：-z（相机前向）→ +x 下俯 20°。"""
  pitch = np.deg2rad(20.0)
  forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
  up = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
  # 列 = 局部轴的世界像：x→right = up × ... 正交右手系构造。
  mat = np.stack([np.cross(up, -forward), up, -forward], axis=1)
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, mat.reshape(9))
  return " ".join(f"{v:.6f}" for v in q)


def _robot_entity() -> EntityCfg:
  xml = ROBOT_XML.replace("@CAM_QUAT@", _cam_quat())
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))


def _crate_verts_faces(n: int = 1) -> tuple[np.ndarray, np.ndarray]:
  return _box_mesh_arrays(CRATE_HALF, n)


def _make(backend: str, num_envs: int, entities, sensors):
  """与 tests/test_classic_raycast.py 同构的场景构建（含 sensor_context 接线）。"""
  from mjlab.scene import Scene

  entity_cfgs = dict(entities)
  entity_cfgs.setdefault("robot", _robot_entity())
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=5.0,
    entities=entity_cfgs,
    sensors=tuple(sensors),
  )
  scene = Scene(scene_cfg, "cpu")
  model = scene.compile()
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


def _build_and_sense(backend: str, num_envs, entities, sensors, steps=3):
  scene, sim = _make(backend, num_envs, entities, sensors)
  sim.reset()
  for _ in range(steps):
    sim.step()
    sim.sense()
  return scene, sim


# 传感器配置。


def _ring_cfg(name="ring_scan", direction=(1.0, 0.0, -0.15), radius=0.12, samples=15):
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=RingPatternCfg(
      rings=(RingPatternCfg.Ring(radius=radius, num_samples=samples),),
      include_center=True,
      direction=direction,
    ),
    max_distance=10.0,
  )


def _pinhole_cfg(name="pinhole_scan", site="cam_site", width=8, height=6):
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="site", name=site, entity="robot"),
    pattern=PinholeCameraPatternCfg(width=width, height=height, fovy=45.0),
    max_distance=10.0,
  )


def _core_ctx(sim, name):
  return sim._sensor_context._raycast_ctxs[name]


# ---------------------------------------------------------------------------
# mesh 命中 vs mj_ray f64 逐射线参考（Ring / Pinhole）。
# ---------------------------------------------------------------------------


def _mj_ray_reference(sim, sensor) -> np.ndarray:
  """逐射线 f64 参考：scratch MjData + mj_kinematics + mj_ray。"""
  model = sim.mj_model
  scratch = mujoco.MjData(model)
  origins = sensor._cached_world_origins.detach().double().numpy()
  dirs = sensor._cached_world_rays.detach().double().numpy()
  B, N, _ = origins.shape
  out = np.full((B, N), -1.0)
  bodyexclude = sensor._frame_infos[0][2] if sensor.cfg.exclude_parent_body else -1
  for b in range(B):
    scratch.qpos[:] = sim.data.qpos[b].detach().cpu().numpy()
    mujoco.mj_kinematics(model, scratch)
    for n in range(N):
      out[b, n] = mujoco.mj_ray(
        model, scratch, origins[b, n], dirs[b, n], None, True, bodyexclude, None
      )
  return out


def _mj_ray_entities():
  """crate（freejoint 动体，初始 45° 旋转）+ pillar（静态 body，constant 模式）。"""
  verts, faces = _crate_verts_faces()
  pv, pf = _box_mesh_arrays(PILLAR_HALF, 1)
  return {
    "crate": _mesh_entity(
      "crate", "crate_body", "crate_geom", "1.4 0 0.55", CRATE_QUAT, verts, faces
    ),
    "pillar": _static_mesh_entity(
      "pillar", "pillar_body", "pillar_geom", "0.7 -0.35 0.55", pv, pf
    ),
  }


@pytest.mark.parametrize("sensor_cfg", [_ring_cfg(), _pinhole_cfg()])
def test_mesh_hit_matches_mj_ray(sensor_cfg):
  """落体 mesh crate 场景：射线距离 vs mj_ray 逐射线 f64，rel < 1e-4。"""
  scene, sim = _build_and_sense("classic", 2, _mj_ray_entities(), (sensor_cfg,))
  sensor = scene[sensor_cfg.name]
  ctx = _core_ctx(sim, sensor_cfg.name)
  assert len(ctx.mesh_colliders) == 2, "crate 与 pillar 两个 mesh 候选"

  distances = sensor.data.distances.detach().double().numpy()
  reference = _mj_ray_reference(sim, sensor)

  hit_core = distances >= 0
  hit_ref = reference >= 0
  assert hit_core.any(), "场景设计应保证有 mesh/plane 命中"
  # 命中/无命中模式完全一致（单面剔除 vs C 端双面：射线均从实体外部发射，等价）。
  assert (hit_core == hit_ref).all(), (
    f"命中模式不一致：core {hit_core.sum()} vs mj_ray {hit_ref.sum()} 射线命中"
  )
  # 命中距离 rel < 1e-4。
  denom = np.abs(reference[hit_ref])
  rel = np.abs(distances[hit_ref] - reference[hit_ref]) / np.maximum(denom, 1e-9)
  assert rel.max() < 1e-4, f"命中距离最大 rel err {rel.max():.3e}"
  # mesh crate 上应有命中（crate 前脸距传感器 ~1.0m，取 < 1.5 区分 plane 远命中）。
  assert (distances[hit_ref] < 1.5).any(), "应有 mesh 命中"


def test_parity_classic_vs_warp_with_mesh():
  """跨后端对齐（F-track 口径）：terrain 场景 + mesh crate 在场，Grid 与 Ring。"""
  verts, faces = _crate_verts_faces()
  terrain = {
    "crate": _mesh_entity(
      "crate", "crate_body", "crate_geom", "0.45 0 0.45", CRATE_QUAT, verts, faces
    ),
  }
  grid = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(1.2, 1.2), resolution=0.08, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )
  ring = _ring_cfg(name="ring_scan", direction=(0.3, 0.0, -1.0))
  dc, _ = _build_and_sense("classic", 4, terrain, (grid, ring))
  dw, _ = _build_and_sense("warp", 4, terrain, (grid, ring))

  for name in ("terrain_scan", "ring_scan"):
    mc = dc[name].data
    mw = dw[name].data
    torch.testing.assert_close(mc.distances, mw.distances, atol=1e-3, rtol=0)
    both = (mc.distances >= 0) & (mw.distances >= 0)
    assert both.any(), f"{name}: 应有共同命中"
    torch.testing.assert_close(
      mc.hit_pos_w[both], mw.hit_pos_w[both], atol=2e-3, rtol=0
    )
    dot = (mc.normals_w[both] * mw.normals_w[both]).sum(-1).clamp(-1, 1)
    ang = torch.acos(dot)
    assert (ang < torch.deg2rad(torch.tensor(5.0))).all(), (
      f"{name}: 法线夹角最大 {ang.max():.4f} rad"
    )
    # mesh crate 上应有命中（顶面距 base ~0.15-1.2m，区分 hfield/plane 远命中）。
    assert (mc.distances[both] < 1.3).any(), f"{name}: 应有 mesh 命中"


# ---------------------------------------------------------------------------
# 语义：无命中 / 三模式端到端 / body 排除。
# ---------------------------------------------------------------------------


def test_no_hit_semantics_non_grid():
  """Ring/Pinhole 指向天空：distances=-1、normals=0、hit_pos=射线起点。"""
  scene, sim = _build_and_sense(
    "classic",
    2,
    _mj_ray_entities(),
    (
      _ring_cfg(name="ring_scan", direction=(0.0, 0.0, 1.0)),
      _pinhole_cfg(name="pinhole_scan", site="up_site", width=4, height=4),
    ),
  )
  for name in ("ring_scan", "pinhole_scan"):
    sensor = scene[name]
    d = sensor.data
    assert (d.distances == -1).all(), name
    assert (d.normals_w == 0).all(), name
    origins = d.frame_pos_w[:, 0, :].unsqueeze(1) + sensor._local_offsets.unsqueeze(0)
    torch.testing.assert_close(d.hit_pos_w, origins, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
  "pattern",
  [
    GridPatternCfg(size=(0.5, 0.5), resolution=0.25, direction=(1.0, 0.0, -0.15)),
    PinholeCameraPatternCfg(width=4, height=4, fovy=45.0),
    RingPatternCfg.single_ring(radius=0.1, num_samples=4, direction=(1.0, 0.0, -0.15)),
  ],
)
def test_classic_patterns_all_work(pattern):
  """R1 合同：三种模式在 classic 下统一走通用核心（Grid 不回归，其余不再 raise）。"""
  # Pinhole 局部 -z 为前向，挂 cam_site（前向 +x 下俯 20°）；Grid/Ring 直接给定
  # 前向下俯方向。场景无地板：命中只能来自 crate/pillar，顺带验证 mesh 候选。
  frame = (
    ObjRef(type="site", name="cam_site", entity="robot")
    if isinstance(pattern, PinholeCameraPatternCfg)
    else ObjRef(type="body", name="base", entity="robot")
  )
  scene, sim = _build_and_sense(
    "classic",
    2,
    _mj_ray_entities(),
    (
      RayCastSensorCfg(
        name="scan",
        frame=frame,
        pattern=pattern,
      ),
    ),
  )
  d = scene["scan"].data
  assert (d.distances >= 0).any(), "前向射线应命中 crate"


def test_mesh_on_parent_body_excluded():
  """mesh geom 挂传感器自身 body：exclude_parent_body 必须排除自身命中。

  机器人 base（z=2）下方挂一个 mesh 立方体障碍（中心 z≈1.55，顶面 z≈1.85）：
  默认排除自身 → 向下射线全部无命中（场景无其他几何）；关闭排除 → 命中立方体
  顶面（≈0.15）。
  """
  verts, faces = _box_mesh_arrays(0.3, 1)
  xml = """
<mujoco model="cube_robot">
  <worldbody>
    <body name="base" pos="0 0 2">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.05 0.05 0.05" mass="5.0"/>
      <geom name="cube_geom" type="mesh" mesh="cube_mesh" pos="0 0 -0.45"
            mass="2.0"/>
    </body>
  </worldbody>
</mujoco>
"""

  def spec_fn():
    spec = mujoco.MjSpec.from_string(xml)
    mesh = spec.add_mesh()
    mesh.name = "cube_mesh"
    mesh.uservert = np.asarray(verts, dtype=np.float32).reshape(-1)
    mesh.userface = np.asarray(faces, dtype=np.int32).reshape(-1)
    return spec

  robot = EntityCfg(spec_fn=spec_fn)
  down_ring = _ring_cfg(name="scan", direction=(0.0, 0.0, -1.0), radius=0.12)

  def make(with_exclusion: bool):
    cfg = down_ring
    if not with_exclusion:
      cfg = dataclasses.replace(cfg, exclude_parent_body=False)
    scene, sim = _build_and_sense("classic", 2, {"robot": robot}, (cfg,))
    return scene["scan"].data

  d_excl = make(with_exclusion=True)
  d_self = make(with_exclusion=False)
  # 排除自身：场景无其他几何（无地板），向下射线应全部无命中。
  assert (d_excl.distances == -1).all(), (
    f"排除自身后不应有命中，max={d_excl.distances.max():.3f}"
  )
  # 不排除：射线命中自身立方体顶面（origin z≈2 - 顶面 z≈1.85 ≈ 0.15）。
  assert (d_self.distances > 0).all()
  assert (d_self.distances < 0.3).all()
  # 顶面命中距离 = frame z - (2 - 0.45 + 0.3)，随自由落体 mm 级漂移，用宽松窗。
  expected = d_self.frame_pos_w[:, 0, 2] - 1.85
  torch.testing.assert_close(
    d_self.distances,
    expected.unsqueeze(1).expand_as(d_self.distances),
    atol=5e-3,
    rtol=0,
  )


# ---------------------------------------------------------------------------
# BVH 遍历 vs 暴力全三角形求交（内部一致性，随机非凸网格）。
# ---------------------------------------------------------------------------


def test_bvh_traversal_matches_bruteforce():
  from mjlab.sensor import raycast_core

  torch.manual_seed(7)
  for trial in range(16):
    nv = int(torch.randint(40, 220, (1,)))
    nf = int(torch.randint(120, 1200, (1,)))
    verts = torch.randn(nv, 3) * 2.0
    faces = torch.randint(0, nv, (nf, 3))
    collider = raycast_core.MeshCollider.from_mesh(verts, faces)

    lo = torch.randn(256, 3) * 3.0
    ld = torch.randn(256, 3)
    ld = ld / ld.norm(dim=-1, keepdim=True)
    t_bvh, n_bvh = collider.closest_hit(lo, ld, 8.0)
    t_bf, n_bf = raycast_core.intersect_trians_bruteforce(verts, faces, lo, ld, 8.0)

    hit_bvh = torch.isfinite(t_bvh)
    hit_bf = torch.isfinite(t_bf)
    assert (hit_bvh == hit_bf).all(), f"trial {trial}: 命中模式不一致"
    if hit_bf.any():
      torch.testing.assert_close(t_bvh[hit_bf], t_bf[hit_bf], atol=1e-4, rtol=0)
      dot = (n_bvh[hit_bf] * n_bf[hit_bf]).sum(-1)
      assert (dot > 0.9999).all(), f"trial {trial}: 法线不一致"


def test_bvh_cache_reused():
  from mjlab.sensor import raycast_core

  verts, faces = _crate_verts_faces()
  tv = torch.from_numpy(verts)
  tf = torch.from_numpy(faces.astype(np.int64))
  c1 = raycast_core.MeshCollider.from_mesh(tv, tf)
  c2 = raycast_core.MeshCollider.from_mesh(tv, tf)
  assert c1.bvh is c2.bvh, "同网格二次构建应命中缓存"


# ---------------------------------------------------------------------------
# 性能门槛：256 envs × 720 rays（Ring）+ 2028 三角 mesh crate。
# ---------------------------------------------------------------------------


def test_perf_lidar_256x720():
  """性能记录与回归线：256 envs × 720 rays（Ring）+ 2028 三角 mesh crate。

  路线图 R1 门槛为 sense ≤ 1ms（min 口径）。实测（Apple M1 Pro，torch CPU，
  2026-09-26）：最优化的通用 CPU 实现（geom AABB 剪枝 + 逐 mesh BVH + 跨环境
  精确去重 + 叶级紧凑）closest_hit min ≈ 16ms，**未达 1ms 门槛**——瓶颈是
  BVH 逐射线下降的逐 op 分发（~33 轮 × ~55 个 eager kernel，与 F-track
  height_scan 的结论同源：eager torch 受 CPU 逐 op 分发限制）。按路线图预案
  （R1：「不达标先调 AABB 剪枝，仍不达标记录数字并在报告标注」），该负载标注
  **「建议走阶段 1 GPU 射线核心」**。

  本测试断言为**回归线**（防止未来改动造成数量级劣化），不是对 1ms 门槛的
  放宽：门槛判定数字、优化过程与结论完整记录于 docs/results/r1-raycast-core.md
  与 docs/results/worklog-R1.md。

  场景：plane 地板 + 动体 mesh crate（12*13^2=2028 三角，前方 2.5m，前向水平
  微俯射线束）——平行射线束整体穿过 crate 截面，几乎所有射线都是 BVH 候选
  （命中率 1.00，lidar 打在障碍物上的真实负载；跨环境去重后实际遍历 720 根
  局部射线）。另打印全 miss 与 12 三角小 mesh 两个上下文数字。

  测量协议（沿用 F-track）：5 块 × 每块 30 次，取全部采样最小值（min 对代码
  确定性成本无噪声；共享机器桌面负载 ±15% 中位数抖动）。
  """
  verts, faces = _crate_verts_faces(n=13)
  assert len(faces) >= 2028
  crate = _mesh_entity(
    "crate", "crate_body", "crate_geom", "2.5 0 0.45", CRATE_QUAT, verts, faces
  )
  scan = _ring_cfg(name="scan", direction=(1.0, 0.0, -0.2), radius=0.1, samples=719)
  scene, sim = _make("classic", 256, {"crate": crate}, (scan,))
  sim.reset()
  for _ in range(3):
    sim.step()
    sim.sense()

  from mjlab.sensor import raycast_cpu

  ctx = _core_ctx(sim, "scan")
  sensor = sim._sensor_context.raycast_sensors[0]
  assert sensor.num_rays == 720
  _, _, origins, directions = raycast_cpu.compute_world_rays(sensor)
  distances, _, _ = ctx.closest_hit(origins, directions)  # warmup
  hit = distances >= 0
  cand = float(hit.float().mean())
  assert hit.any(), "射线束应命中 crate"

  def sense_pass():
    _, _, o, d = raycast_cpu.compute_world_rays(sensor)
    dist, norm, _ = ctx.closest_hit(o, d)
    raycast_cpu.finalize(sensor, dist, norm, o, d)

  def block_raw(fn, n=30):
    times = []
    gc_was = gc.isenabled()
    gc.disable()
    try:
      for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    finally:
      if gc_was:
        gc.enable()
    return times

  samples = []
  for _ in range(5):
    samples.extend(block_raw(lambda: ctx.closest_hit(origins, directions)))
  best = min(samples)
  med = float(np.median(samples))
  sense_best = min(block_raw(sense_pass))
  print(
    f"\nclassic closest_hit @B=256,N=720,2028-tri mesh: min {best:.3f} ms "
    f"(median {med:.3f}, n={len(samples)}); 命中率 {cand:.2f}; "
    f"完整 sense pass min {sense_best:.3f} ms"
  )
  print(
    "路线图 R1 门槛 sense ≤ 1ms 未达（torch CPU 逐 op 分发受限）→ "
    "标注：建议走阶段 1 GPU 射线核心（见 docs/results/r1-raycast-core.md）"
  )
  # 回归线：实测 min ≈ 16ms（含 ±15% 桌面负载抖动与留量），防止数量级劣化。
  assert best <= 25.0, f"closest_hit {best:.3f} ms 超出回归线 25ms"
