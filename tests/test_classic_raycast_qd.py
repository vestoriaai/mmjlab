"""classic 后端 qd raycast 后端（raycast_backend="qd"/auto→qd）：语义与对齐。

对齐合同（固定，不许放宽；与 1A 同口径）：
- 三模式（Grid/Pinhole/Ring）× 三类场景（图元/mesh/hfield），qd 后端 vs
  ``mujoco.mj_ray`` 逐射线 f64 oracle：命中/无命中模式一致（测度零退化按
  1A 分类法豁免并计数），命中距离 rel ≤ 1e-4，法线 |dot| ≥ 0.999（符号
  翻转记录不判负——qd mesh 法线朝 -d，内部起点出射时与 mj_ray 反向）。
- 跨后端：qd vs torch 同场景同射线，distances atol=1e-3，共同命中
  hit_pos atol=2e-3，法线夹角 < 5°（沿用 R1/F-track 口径）。
- 语义：无命中 -1/0/origin；max_distance 裁剪；逐帧 bodyexclude。
- 回退：raycast_backend="torch" 行为不变；auto 在无 quadrants 环境回退。

性能（1B 门槛，require_quiet_machine 协议，min-of-N）：
- lidar 门槛负载（256×720 Ring + 2028 三角 mesh，不去重）qd.metal e2e
  ≤ 2ms（e2e = 位姿上传 + launch + sync + 输出交付，见
  docs/results/stage1b-report.md 的口径说明）。
- 静态去重口径（同负载、跨环境精确去重）≤ 1ms。
"""

from __future__ import annotations

import dataclasses
import gc
import time

import mujoco
import numpy as np
import pytest
import torch
from test_classic_raycast_core import (
  CRATE_HALF,
  CRATE_QUAT,
  _box_mesh_arrays,
  _cam_quat,
  _mesh_entity,
  _robot_entity,
  _static_mesh_entity,
)

from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
  RingPatternCfg,
  raycast_qd,
)
from mjlab.sensor.raycast_core import RaycastCoreContext
from mjlab.sensor.raycast_qd import QdRaycastContext
from mjlab.sim.sim import SimulationCfg, make_simulation

# qd 后端依赖可选的 quadrants/qd-render-poc：未安装时整文件跳过（回退路径由
# 其余 raycast 测试在无 qd 环境下天然覆盖）。
pytestmark = pytest.mark.skipif(
  not raycast_qd.is_available(),
  reason="quadrants/qd-render-poc 未安装（可选依赖）",
)

# ---------------------------------------------------------------------------
# 场景。
# ---------------------------------------------------------------------------

# 图元场景：plane + 五类解析图元（sphere/capsule/cylinder/box/ellipsoid），
# 体位姿含 15°/35° 与 77° 旋转（对齐 1A T2/T4 的旋转覆盖）。
PRIM_XML = """
<mujoco model="qd_prim">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="sph_b" pos="1.2 0 0.5" quat="0.9659 0 0.2588 0">
      <geom name="sph" type="sphere" size="0.25"/>
    </body>
    <body name="cap_b" pos="1.2 1.0 0.4" quat="0.7071 0.3 0.3 0.5577">
      <geom name="cap" type="capsule" size="0.12 0.35"/>
    </body>
    <body name="cyl_b" pos="1.2 -1.0 0.45" quat="0.8 0.2 0.4 0.4">
      <geom name="cyl" type="cylinder" size="0.15 0.3"/>
    </body>
    <body name="box_b" pos="-1.5 0.6 0.35" quat="0.9 0.1 0.2 0.37">
      <geom name="box" type="box" size="0.25 0.18 0.12"/>
    </body>
    <body name="ell_b" pos="-1.5 -0.6 0.5" quat="0.75 0.33 0.33 0.46">
      <geom name="ell" type="ellipsoid" size="0.3 0.2 0.15"/>
    </body>
  </worldbody>
</mujoco>
"""

HFIELD_XML = """
<mujoco model="qd_terrain">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="hill_body" pos="0 0 0">
      <geom name="hill" type="hfield" hfield="hf"/>
    </body>
  </worldbody>
  <asset>
    <hfield name="hf" nrow="48" ncol="48" size="6 6 0.5 0.02"/>
  </asset>
</mujoco>
"""


GHOST_ROBOT_XML = """
<mujoco model="qd_ghost_robot">
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <freejoint name="free_joint"/>
      <inertial pos="0 0 0" mass="5.0" diaginertia="1 1 1"/>
      <site name="cam_site" pos="0 0 0" quat="@CAM_QUAT@"/>
      <site name="up_site" pos="0 0 0" quat="0 1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ghost_robot_entity() -> EntityCfg:
  """无 geom 的机器人（freejoint + sites）：跨后端对齐/裁剪测试用，
  避免 qd 独有的 base box 命中混入 mesh 对拍。"""
  xml = GHOST_ROBOT_XML.replace("@CAM_QUAT@", _cam_quat())
  return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))


def _prim_entities() -> dict[str, EntityCfg]:
  return {"prim": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(PRIM_XML))}


def _mesh_entities() -> dict[str, EntityCfg]:
  verts, faces = _box_mesh_arrays(CRATE_HALF, 13)  # 2028 tri
  pv, pf = _box_mesh_arrays(0.15, 1)
  return {
    "crate": _mesh_entity(
      "crate", "crate_body", "crate_geom", "1.4 0 0.55", CRATE_QUAT, verts, faces
    ),
    "pillar": _static_mesh_entity(
      "pillar", "pillar_body", "pillar_geom", "0.7 -0.35 0.55", pv, pf
    ),
  }


def _terrain_entities() -> dict[str, EntityCfg]:
  return {"terrain": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(HFIELD_XML))}


def _fill_hfield(model: mujoco.MjModel) -> None:
  nrow, ncol = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
  r = np.linspace(0.0, 1.0, nrow)[:, None]
  c = np.linspace(0.0, 1.0, ncol)[None, :]
  h = 0.25 * (0.5 + 0.5 * np.sin(6.0 * r) * np.cos(5.0 * c))
  h += 0.05 * np.sin(20.0 * r) * np.sin(18.0 * c)
  model.hfield_data[:] = h.reshape(-1)


def _make(num_envs: int, entities, sensors):
  """classic 场景构建（含 sensor_context 接线），与既有测试同构。"""
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
  if model.nhfield:
    _fill_hfield(model)
  sim = make_simulation(
    num_envs=num_envs,
    cfg=SimulationCfg(backend="classic", njmax=20),
    model=model,
    device="cpu",
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  if scene.sensor_context is not None:
    sim.set_sensor_context(scene.sensor_context)
  return scene, sim


def _build_and_sense(num_envs, entities, sensors, steps=3):
  scene, sim = _make(num_envs, entities, sensors)
  sim.reset()
  for _ in range(steps):
    sim.step()
    sim.sense()
  return scene, sim


def _base_frame() -> ObjRef:
  return ObjRef(type="body", name="base", entity="robot")


def _patterns(name: str) -> list[RayCastSensorCfg]:
  """三模式（全部显式 raycast_backend="qd"）。"""
  return [
    _qd_cfg(
      name,
      GridPatternCfg(size=(0.6, 0.6), resolution=0.15, direction=(1.0, 0.0, -0.3)),
      _base_frame(),
    ),
    _qd_cfg(
      name,
      PinholeCameraPatternCfg(width=8, height=6, fovy=45.0),
      ObjRef(type="site", name="cam_site", entity="robot"),
    ),
    _qd_cfg(
      name,
      RingPatternCfg.single_ring(
        radius=0.12, num_samples=15, direction=(1.0, 0.0, -0.35)
      ),
      _base_frame(),
    ),
  ]


def _qd_cfg(name: str, pattern, frame: ObjRef) -> RayCastSensorCfg:
  return RayCastSensorCfg(name=name, frame=frame, pattern=pattern, raycast_backend="qd")


# ---------------------------------------------------------------------------
# mj_ray f64 oracle + 退化分类（1A 分类法）。
# ---------------------------------------------------------------------------


def _mesh_edges(model: mujoco.MjModel, gid: int, gpos, gmat) -> np.ndarray:
  """mesh geom 的世界系棱段集 (E, 2, 3)。"""
  hid = int(model.geom_dataid[gid])
  vadr, vnum = int(model.mesh_vertadr[hid]), int(model.mesh_vertnum[hid])
  fadr, fnum = int(model.mesh_faceadr[hid]), int(model.mesh_facenum[hid])
  verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
  faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int64)
  tri = verts[faces]
  edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
  return edges @ gmat.T + gpos


def _point_seg_dist(p: np.ndarray, edges: np.ndarray) -> float:
  a = edges[:, 0]
  ab = edges[:, 1] - a
  t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
  proj = a + t[:, None] * ab
  return float(np.linalg.norm(p - proj, axis=-1).min())


def _degenerate_reason(
  model: mujoco.MjModel,
  gid: int,
  gpos: np.ndarray,
  gmat: np.ndarray,
  o: np.ndarray,
  d: np.ndarray,
  t: float,
  mesh_edge_cache: dict[int, np.ndarray],
) -> str | None:
  """1A 分类法：测度零退化（切线/棱/边界/网格线）命中返回原因，否则 None。"""
  gtype = int(model.geom_type[gid])
  size = np.asarray(model.geom_size[gid], dtype=np.float64)
  p = o + t * d
  lp = gmat.T @ (p - gpos)  # geom 局部系命中点
  tol = 1e-5
  PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
  SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
  CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
  CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
  BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
  MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
  HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)
  if gtype == PLANE:
    sx, sy = float(size[0]), float(size[1])
    if sx > 0 and abs(abs(lp[0]) - sx) < tol:
      return "plane-x-edge"
    if sy > 0 and abs(abs(lp[1]) - sy) < tol:
      return "plane-y-edge"
    return None
  if gtype == SPHERE:
    perp = float(np.linalg.norm(np.cross(gpos - o, d)))
    if abs(perp - float(size[0])) < 1e-4:
      return "sphere-tangent"
    return None
  if gtype == CAPSULE:
    axis = gmat[:, 2] * float(size[1])
    p0, p1 = gpos - axis, gpos + axis
    ab = p1 - p0
    ts = np.clip(np.dot(o - p0, ab) / max(np.dot(ab, ab), 1e-30), 0.0, 1.0)
    perp = float(np.linalg.norm(np.cross(p0 + ts * ab - o, d)))
    if abs(perp - float(size[0])) < 1e-4:
      return "capsule-tangent"
    return None
  if gtype == ELLIPSOID:
    return None  # 无简单退化特征（1A 在 ellipsoid 上 rel 门内全过）
  if gtype == CYLINDER:
    axis = gmat[:, 2]
    ts = np.dot(axis, gpos - o) / max(np.dot(axis, axis), 1e-30)
    perp = float(np.linalg.norm(np.cross(axis, o + ts * d - gpos)))
    if abs(perp - float(size[0])) < 1e-4:
      return "cylinder-tangent"
    if abs(abs(lp[2]) - float(size[1])) < tol:
      if abs(np.hypot(lp[0], lp[1]) - float(size[0])) < tol:
        return "cylinder-rim"
    return None
  if gtype == BOX:
    near = sum(1 for i in range(3) if abs(abs(lp[i]) - float(size[i])) < tol)
    if near >= 2:
      return "box-edge"
    return None
  if gtype == MESH:
    if gid not in mesh_edge_cache:
      mesh_edge_cache[gid] = _mesh_edges(model, gid, gpos, gmat)
    if _point_seg_dist(p, mesh_edge_cache[gid]) < 1e-6:
      return "mesh-edge-graze"
    return None
  if gtype == HFIELD:
    sx, sy = float(size[0]), float(size[1])
    hid = int(model.geom_dataid[gid])
    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    dx, dy = 2 * sx / (ncol - 1), 2 * sy / (nrow - 1)
    u, v = lp[0] + sx, lp[1] + sy
    if abs(u / dx - round(u / dx)) * dx < tol or abs(v / dy - round(v / dy)) * dy < tol:
      return "hfield-grid-line"
    if abs(abs(lp[0]) - sx) < tol or abs(abs(lp[1]) - sy) < tol:
      return "hfield-side-wall"
    return None
  return None


def _qd_oracle_check(
  sim,
  sensor,
  sample: int = 128,
  seed: int = 0,
  rel_tol: float = 1e-4,
  dot_tol: float = 0.999,
) -> dict:
  """逐射线 mj_ray f64 oracle 对齐检查（含退化豁免桶）。"""
  model = sim.mj_model
  ctx = sim._sensor_context._raycast_ctxs[sensor.cfg.name]
  origins = sensor._cached_world_origins.detach().double().numpy()
  dirs = sensor._cached_world_rays.detach().double().numpy()
  dist = sensor.data.distances.detach().double().numpy()
  normals = sensor.data.normals_w.detach().double().numpy()
  B, N, _ = origins.shape
  R = sensor.num_rays_per_frame
  xpos = sim.data.geom_xpos.detach().double().numpy().reshape(B, -1, 3)
  xmat = sim.data.geom_xmat.detach().double().numpy().reshape(B, -1, 9)
  G = model.ngeom

  rng = np.random.default_rng(seed)
  idx = rng.choice(B * N, size=min(sample, B * N), replace=False)
  geomid_buf = np.zeros(1, dtype=np.int32)
  normal_buf = np.zeros(3)
  scratch = mujoco.MjData(model)
  mujoco.mj_kinematics(model, scratch)
  mesh_edge_cache: dict[int, np.ndarray] = {}

  n_mismatch = 0
  n_exempt = 0
  exempt_reasons: list[str] = []
  rels: list[float] = []
  dots: list[float] = []
  flips = 0
  failures: list[tuple] = []

  def classify(gid, o, d, t):
    gpos = np.asarray(scratch.geom_xpos[gid]).copy()
    gmat = np.asarray(scratch.geom_xmat[gid]).reshape(3, 3).copy()
    return _degenerate_reason(model, gid, gpos, gmat, o, d, t, mesh_edge_cache)

  for t_ in idx:
    b, n = divmod(int(t_), N)
    scratch.geom_xpos[:] = xpos[b].reshape(G, 3)
    scratch.geom_xmat[:] = xmat[b].reshape(G, 9)
    o = origins[b, n]
    d = dirs[b, n]
    d = d / np.linalg.norm(d)
    bodyexclude = ctx._frame_body_exclude[n // R]
    ref = mujoco.mj_ray(
      model, scratch, o, d, None, True, bodyexclude, geomid_buf, normal_buf
    )
    # qd kernel 在 max_distance 处裁剪（warp 路径宿主侧同语义），oracle 对齐
    # 参考需施加同一裁剪。
    if ref > ctx._max_distance:
      ref = -1.0
    gd, gn = dist[b, n], normals[b, n]
    ref_hit, qd_hit = ref >= 0, gd >= 0
    if ref_hit != qd_hit:
      # 分歧：仅当（任一侧的）mj_ray 侧命中落在退化特征上才豁免。
      reason = None
      if ref_hit:
        reason = classify(int(geomid_buf[0]), o, d, ref)
      if reason:
        n_exempt += 1
        exempt_reasons.append(reason)
      else:
        n_mismatch += 1
        failures.append((t_, "pattern", float(gd), float(ref)))
      continue
    if not ref_hit:
      continue
    gid = int(geomid_buf[0])
    rel = abs(gd - ref) / max(abs(ref), 1e-9)
    dot = float(np.dot(gn, normal_buf))
    if rel > rel_tol:
      reason = classify(gid, o, d, ref)
      if reason:
        n_exempt += 1
        exempt_reasons.append(reason)
        continue
      n_mismatch += 1
      failures.append((t_, "rel", float(gd), float(ref)))
      continue
    rels.append(rel)
    if abs(dot) < dot_tol:
      reason = classify(gid, o, d, ref)
      if reason:
        n_exempt += 1
        exempt_reasons.append(reason)
        continue
      n_mismatch += 1
      failures.append((t_, "normal", float(dot), float(ref)))
      continue
    dots.append(abs(dot))
    if dot < 0:
      flips += 1

  return {
    "n": len(idx),
    "max_rel": max(rels) if rels else float("nan"),
    "min_absdot": min(dots) if dots else float("nan"),
    "flips": flips,
    "mismatch": n_mismatch,
    "exempt": n_exempt,
    "exempt_reasons": exempt_reasons,
    "failures": failures,
  }


# ---------------------------------------------------------------------------
# 对齐：三模式 × 图元 / mesh / hfield。
# ---------------------------------------------------------------------------

_SEEDS = {"primitives": 11, "mesh": 22, "hfield": 33}


@pytest.mark.parametrize("mode_idx", [0, 1, 2])
@pytest.mark.parametrize("scene_name", ["primitives", "mesh", "hfield"])
def test_qd_aligns_mj_ray(scene_name: str, mode_idx: int):
  """qd 后端三模式 × 三类场景 vs mj_ray 逐射线 f64 oracle。"""
  entities = {
    "primitives": _prim_entities,
    "mesh": _mesh_entities,
    "hfield": _terrain_entities,
  }[scene_name]()
  cfg = _patterns("scan")[mode_idx]
  scene, sim = _build_and_sense(4, entities, (cfg,))
  sensor = scene["scan"]
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], QdRaycastContext)
  assert (sensor.data.distances >= 0).any(), "场景设计应保证有命中"
  stats = _qd_oracle_check(sim, sensor, sample=128, seed=_SEEDS[scene_name] + mode_idx)
  print(
    f"\n[{scene_name}/{cfg.pattern.__class__.__name__}] n={stats['n']} "
    f"max_rel={stats['max_rel']:.2e} min|dot|={stats['min_absdot']:.6f} "
    f"flips={stats['flips']} exempt={stats['exempt']} "
    f"({sorted(set(stats['exempt_reasons']))})"
  )
  assert not stats["failures"], f"对齐失败: {stats['failures'][:4]}"
  assert not stats["mismatch"]
  if not np.isnan(stats["max_rel"]):
    assert stats["max_rel"] <= 1e-4, f"rel {stats['max_rel']:.2e} > 1e-4"


# ---------------------------------------------------------------------------
# 跨后端：qd vs torch。
# ---------------------------------------------------------------------------


def test_cross_backend_qd_vs_torch():
  """同场景同射线：qd vs torch distances 一致（既有口径 1e-3/2e-3/5°）。"""
  # 网格偏移避开 pillar 面 y=-0.5（共面射线是测度零退化，torch/qd 对拍会分歧）
  grid = GridPatternCfg(size=(0.9, 0.9), resolution=0.1, direction=(0.3, 0.0, -1.0))
  ring = RingPatternCfg.single_ring(
    radius=0.12, num_samples=15, direction=(1.0, 0.0, -0.5)
  )
  torch_cfgs = (
    RayCastSensorCfg(
      name="grid_scan", frame=_base_frame(), pattern=grid, raycast_backend="torch"
    ),
    RayCastSensorCfg(
      name="ring_scan", frame=_base_frame(), pattern=ring, raycast_backend="torch"
    ),
  )
  qd_cfgs = (
    RayCastSensorCfg(
      name="grid_scan", frame=_base_frame(), pattern=grid, raycast_backend="qd"
    ),
    RayCastSensorCfg(
      name="ring_scan", frame=_base_frame(), pattern=ring, raycast_backend="qd"
    ),
  )
  entities = dict(_mesh_entities())
  entities["robot"] = _ghost_robot_entity()
  dt, _ = _build_and_sense(4, entities, torch_cfgs)
  dq, _ = _build_and_sense(4, entities, qd_cfgs)

  for name in ("grid_scan", "ring_scan"):
    mt = dt[name].data
    mq = dq[name].data
    torch.testing.assert_close(mq.distances, mt.distances, atol=1e-3, rtol=0)
    both = (mq.distances >= 0) & (mt.distances >= 0)
    assert both.any(), f"{name}: 应有共同命中"
    torch.testing.assert_close(
      mq.hit_pos_w[both], mt.hit_pos_w[both], atol=2e-3, rtol=0
    )
    dot = (mq.normals_w[both] * mt.normals_w[both]).sum(-1).clamp(-1, 1)
    ang = torch.acos(dot)
    assert (ang < torch.deg2rad(torch.tensor(5.0))).all(), (
      f"{name}: 法线夹角最大 {ang.max():.4f} rad"
    )
    assert (mq.distances[both] < 1.3).any(), f"{name}: 应有 mesh 命中"


# ---------------------------------------------------------------------------
# 语义。
# ---------------------------------------------------------------------------


def test_qd_no_hit_semantics():
  """指向天空：distances=-1、normals=0、hit_pos=射线起点。"""
  ring = RingPatternCfg.single_ring(
    radius=0.1, num_samples=8, direction=(0.0, 0.0, 1.0)
  )
  cfgs = (
    _qd_cfg("ring_scan", ring, _base_frame()),
    _qd_cfg(
      "pinhole_scan",
      PinholeCameraPatternCfg(width=4, height=4, fovy=45.0),
      ObjRef(type="site", name="up_site", entity="robot"),
    ),
  )
  scene, _ = _build_and_sense(2, _mesh_entities(), cfgs)
  for name in ("ring_scan", "pinhole_scan"):
    sensor = scene[name]
    d = sensor.data
    assert (d.distances == -1).all(), name
    assert (d.normals_w == 0).all(), name
    origins = d.frame_pos_w[:, 0, :].unsqueeze(1) + sensor._local_offsets.unsqueeze(0)
    torch.testing.assert_close(d.hit_pos_w, origins, atol=1e-5, rtol=0)


def test_qd_max_distance_clip():
  """命中超出 max_distance → -1（kernel 内裁剪，warp 路径同语义）。"""
  near = RayCastSensorCfg(
    name="scan",
    frame=_base_frame(),
    pattern=RingPatternCfg.single_ring(
      radius=0.05, num_samples=8, direction=(1.0, 0.0, -0.18)
    ),
    max_distance=0.5,  # 45° yaw crate 前角伸到 x~0.83（t~0.8-0.92），0.5 全裁剪
    raycast_backend="qd",
  )
  far = dataclasses.replace(near, name="scan_far", max_distance=10.0)
  entities = dict(_mesh_entities())
  entities["robot"] = _ghost_robot_entity()
  scene, _ = _build_and_sense(2, entities, (near, far))
  # 旋转 crate 前角 t~0.8-0.92：max_distance=0.5 应全裁剪，10.0 应有命中
  assert (scene["scan"].data.distances == -1).all()
  assert (scene["scan_far"].data.distances >= 0).any()


def test_qd_parent_body_exclusion():
  """mesh 挂传感器自身 body：排除→无命中；不排除→命中顶面（~0.15）。"""
  verts, faces = _box_mesh_arrays(0.3, 1)
  xml = """
<mujoco model="qd_cube_robot">
  <worldbody>
    <body name="base" pos="0 0 2">
      <freejoint name="free_joint"/>
      <inertial pos="0 0 0" mass="5.0" diaginertia="1 1 1"/>
      <site name="top_site" pos="0 0 0"/>
      <geom name="cube_geom" type="mesh" mesh="cube_mesh" pos="0 0 -0.45"/>
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

  def make(with_exclusion: bool):
    cfg = RayCastSensorCfg(
      name="scan",
      frame=ObjRef(type="site", name="top_site", entity="robot"),
      pattern=RingPatternCfg.single_ring(
        radius=0.12, num_samples=8, direction=(0.0, 0.0, -1.0)
      ),
      raycast_backend="qd",
    )
    if not with_exclusion:
      cfg = dataclasses.replace(cfg, exclude_parent_body=False)
    scene, _ = _build_and_sense(2, {"robot": robot}, (cfg,))
    return scene["scan"].data

  d_excl = make(with_exclusion=True)
  assert (d_excl.distances == -1).all(), (
    f"排除自身后不应有命中，max={d_excl.distances.max():.3f}"
  )
  d_self = make(with_exclusion=False)
  # 不排除：射线命中自身立方体顶面（origin z≈2 - 顶面 z≈1.85 ≈ 0.15）。
  assert (d_self.distances > 0).all()
  assert (d_self.distances < 0.3).all()
  expected = d_self.frame_pos_w[:, 0, 2] - 1.85
  torch.testing.assert_close(
    d_self.distances,
    expected.unsqueeze(1).expand_as(d_self.distances),
    atol=5e-3,
    rtol=0,
  )


def test_qd_multiframe_body_exclusion():
  """双帧（两个体的 site）：逐帧 bodyexclude——各帧射线不命中自身 body。

  A(0,0,2) 与 B(0,0,1) 各挂一个 box；两帧 ray 均 -z 向下。
  A 帧射线穿过 A 自身 box（已排除）→ 命中 B 的 box 顶（t≈0.95）；
  B 帧射线穿过 B 自身 box（已排除）→ 命中地板（t≈0.95）。
  """
  xml_a = """
<mujoco model="qd_box_a">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <body name="body_a" pos="0 0 2">
      <freejoint name="ja"/>
      <geom name="box_a" type="box" size="0.1 0.1 0.1" mass="1.0"/>
      <site name="site_a" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
  xml_b = """
<mujoco model="qd_box_b">
  <worldbody>
    <body name="body_b" pos="0 0 1">
      <freejoint name="jb"/>
      <geom name="box_b" type="box" size="0.1 0.1 0.1" mass="1.0"/>
      <site name="site_b" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
  entities = {
    "robot_a": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml_a)),
    "robot_b": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml_b)),
    "robot": _ghost_robot_entity(),  # 顶掉默认 robot 实体（其 base box 会混入）
  }
  cfg = RayCastSensorCfg(
    name="scan",
    frame=(
      ObjRef(type="site", name="site_a", entity="robot_a"),
      ObjRef(type="site", name="site_b", entity="robot_b"),
    ),
    pattern=GridPatternCfg(
      size=(0.05, 0.05), resolution=0.05, direction=(0.0, 0.0, -1.0)
    ),
    raycast_backend="qd",
  )
  scene, _ = _build_and_sense(2, entities, (cfg,))
  d = scene["scan"].data
  N = d.distances.shape[1] // 2
  frame_a, frame_b = d.distances[:, :N], d.distances[:, N:]
  assert (frame_a > 0.5).all() and (frame_a < 1.2).all(), frame_a
  assert (frame_b > 0.5).all() and (frame_b < 1.2).all(), frame_b


# ---------------------------------------------------------------------------
# 后端选择与回退。
# ---------------------------------------------------------------------------


def test_backend_selection_policy():
  """auto：plane/hfield 场景保持 torch（快速路径 + 斜射线 raise 语义）；
  含 mesh/图元场景选 qd；显式 "qd"/"torch" 生效。"""
  grid = GridPatternCfg(size=(0.4, 0.4), resolution=0.2)
  scene, sim = _make(
    2,
    _terrain_entities(),
    (RayCastSensorCfg(name="scan", frame=_base_frame(), pattern=grid),),
  )
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], RaycastCoreContext)

  scene, sim = _make(
    2,
    _mesh_entities(),
    (RayCastSensorCfg(name="scan", frame=_base_frame(), pattern=grid),),
  )
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], QdRaycastContext)

  scene, sim = _make(
    2,
    _mesh_entities(),
    (
      RayCastSensorCfg(
        name="scan", frame=_base_frame(), pattern=grid, raycast_backend="torch"
      ),
    ),
  )
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], RaycastCoreContext)

  # 显式 qd 在 plane/hfield 场景也可用（hfield 斜射线解锁）。
  scene, sim = _make(
    2,
    _terrain_entities(),
    (
      RayCastSensorCfg(
        name="scan", frame=_base_frame(), pattern=grid, raycast_backend="qd"
      ),
    ),
  )
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], QdRaycastContext)


def test_auto_falls_back_without_quadrants(monkeypatch):
  """auto 在无 quadrants/qd_render_poc 环境回退 torch（模拟 import 失败）。"""
  monkeypatch.setattr(raycast_qd, "is_available", lambda: False)
  grid = GridPatternCfg(size=(0.4, 0.4), resolution=0.2)
  cfg = RayCastSensorCfg(name="scan", frame=_base_frame(), pattern=grid)
  scene, sim = _make(2, _mesh_entities(), (cfg,))
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], RaycastCoreContext)


def test_qd_backend_falls_back_when_init_fails(monkeypatch):
  """raycast_backend="qd" 但 qd 初始化失败：warn + 回退 torch。"""
  monkeypatch.setattr(raycast_qd, "_ensure_init", lambda: False)
  grid = GridPatternCfg(size=(0.4, 0.4), resolution=0.2)
  cfg = RayCastSensorCfg(
    name="scan", frame=_base_frame(), pattern=grid, raycast_backend="qd"
  )
  with pytest.warns(RuntimeWarning, match="falling back to torch"):
    scene, sim = _make(2, _mesh_entities(), (cfg,))
  assert isinstance(sim._sensor_context._raycast_ctxs["scan"], RaycastCoreContext)


# ---------------------------------------------------------------------------
# 性能门槛（T4；require_quiet_machine 协议，min-of-N，门槛不许放宽）。
# ---------------------------------------------------------------------------


def _lidar_setup(num_envs: int):
  """lidar 门槛负载：256×720 Ring + 2028 三角 mesh crate（真实口径，不去重）。"""
  verts, faces = _box_mesh_arrays(CRATE_HALF, 13)
  assert len(faces) >= 2028
  crate = _mesh_entity(
    "crate", "crate_body", "crate_geom", "2.5 0 0.45", CRATE_QUAT, verts, faces
  )
  scan = RayCastSensorCfg(
    name="scan",
    frame=_base_frame(),
    pattern=RingPatternCfg.single_ring(
      radius=0.1, num_samples=719, direction=(1.0, 0.0, -0.2)
    ),
    max_distance=10.0,
    raycast_backend="qd",
  )
  scene, sim = _make(num_envs, {"crate": crate}, (scan,))
  sim.reset()
  for _ in range(3):
    sim.step()
    sim.sense()
  ctx = sim._sensor_context._raycast_ctxs["scan"]
  sensor = sim._sensor_context.raycast_sensors[0]
  assert isinstance(ctx, QdRaycastContext)
  assert sensor.num_rays == 720
  return scene, sim, ctx, sensor


def _time_best(fn, blocks=5, per_block=30):
  samples = []
  gc_was = gc.isenabled()
  gc.disable()
  try:
    for _ in range(blocks):
      for _ in range(per_block):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
  finally:
    if gc_was:
      gc.enable()
  return min(samples)


def test_perf_lidar_qd_metal_gate():
  """qd.metal e2e ≤ 2ms @256×720 + 2028 tri（T3.1/T3.2 优化后）。

  e2e 口径 = 位姿上传 + launch + sync + 输出交付（DLPack 直通 + .cpu()），
  对齐 1A 的「上传+launch+sync+dist/normals 全回读」（回读以直通替代）；
  射线生成在 kernel 内（1A 建议的 0.9ms 上传项消除）。
  """
  from conftest import require_quiet_machine

  require_quiet_machine()
  arch = raycast_qd.qd_arch()
  if arch != "metal":
    pytest.skip(f"Metal 门槛仅在有 metal arch 时断言（当前 {arch}）")
  _, sim, ctx, sensor = _lidar_setup(256)
  distances, _ = ctx.closest_hit_pattern(sensor, dedup=False)  # warmup
  hit = distances >= 0
  assert hit.any(), "射线束应命中 crate"
  print(f"\n命中率 {float(hit.float().mean()):.2f}")

  best = _time_best(lambda: ctx.closest_hit_pattern(sensor, dedup=False))
  print(f"qd.metal closest_hit_pattern (no dedup) min {best:.3f} ms")
  assert best <= 2.0, f"qd.metal e2e {best:.3f} ms 超出 2ms 门槛"


def test_perf_lidar_qd_dedup_gate():
  """去重口径 ≤ 1ms：跨环境平移等价 → 单 world launch + 平铺。

  mjlab 环境按网格平移布局：全动态场景（crate+robot 均随体）下各 env 的
  帧位姿/几何位姿逐 env 相差同一常量平移 → 命中逐 env 相同（f32 舍入内），
  以 env-0 单 world launch 服务全批（含上传跳过：state 未变时不重复上传）。"""
  from conftest import require_quiet_machine

  require_quiet_machine()
  if raycast_qd.qd_arch() not in ("metal", "cuda", "vulkan"):
    pytest.skip("去重门槛在 GPU arch 上断言（qd.cpu 绝对值另行记录）")
  _, sim, ctx, sensor = _lidar_setup(256)
  best = _time_best(lambda: ctx.closest_hit_pattern(sensor, dedup=True))
  print(
    f"qd.{raycast_qd.qd_arch()} closest_hit_pattern (exact dedup) min {best:.3f} ms"
  )
  assert best <= 1.0, f"去重口径 {best:.3f} ms 超出 1ms 门槛"
