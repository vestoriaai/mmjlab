"""classic 后端 camera 渲染后端 = mjwarp CPU 批量渲染（阶段 0）。

对齐标准（写死，不放宽）：
- RGB vs tests/golden/*.npz（mujoco.Renderer/mjr_ 参考）：
  全帧 uint8 平均绝对差 ≤ 2，且 99% 像素（逐像素取三通道最大差）≤ 4；
- depth vs golden（mjr enable_depth_rendering 的 metric 平面深度）：
  命中像素 atol=1e-2, rtol=1e-2（上游 render_test 同口径）；
  背景集合一致（warp 无命中=0 ⟺ mjr 背景= zfar 哨兵值）。

性能门槛（Step 5）：
- 同步 mjwarp 渲染 64 envs×84×84 ≤ 70ms（多进程分片路径，os.cpu_count()<8 时 skip）；
- 与 async_render=True 组合：sense() <1ms，渲染吞吐 ≥ 同步 mjwarp 路径 80%。

golden 场景生成：tests/golden/generate_golden.py（mjr 参考 + 场景/状态自包含）。
"""

from __future__ import annotations

import os
import pathlib
import time

import mujoco
import numpy as np
import pytest
import torch

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"
GOLDEN_NPZ = sorted(GOLDEN_DIR.glob("stage0_*.npz"))

# 轻量场景：形状/dtype/确定性/后端选择用（顶视相机 + freejoint 红盒 + 地板）。
SCENE_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
          rgba="0.5 0.5 0.5 1"/>
    <camera name="overhead_cam" pos="0 0 3" quat="1 0 0 0"
            fovy="45" resolution="84 84"/>
    <body name="floater" pos="0 0 0.5">
      <freejoint name="floater_joint"/>
      <geom name="red_box" type="box" size="0.2 0.2 0.2"
            rgba="1 0 0 1" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make(
  num_envs: int = 4,
  height: int = 84,
  width: int = 84,
  render_backend: str = "mjwarp",
  data_types: tuple[str, ...] = ("rgb",),
  async_render: bool = False,
  xml: str = SCENE_XML,
  extra_cam_cfg: dict | None = None,
):
  from mjlab.entity import EntityCfg
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sensor import CameraSensorCfg
  from mjlab.sim.sim import SimulationCfg, make_simulation

  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=width,
    height=height,
    data_types=data_types,
    render_backend=render_backend,
    async_render=async_render,
    **(extra_cam_cfg or {}),
  )
  entities = {"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))}
  scene = Scene(
    SceneCfg(num_envs=num_envs, env_spacing=5.0, entities=entities, sensors=(cam_cfg,)),
    "cpu",
  )
  model = scene.compile()
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


# ---------------------------------------------------------------------------
# Step 2a: golden 容差对齐（RGB + depth）。
# ---------------------------------------------------------------------------


def _apply_golden_visuals(mjm: mujoco.MjModel) -> None:
  """Reapply the golden scenes' visual settings post-compile.

  mjlab's spec compile drops the XML ``<visual>`` element (headlight back on,
  GL MSAA back to 4x); these two settings are load-bearing for tight
  cross-renderer alignment (see tests/golden/generate_golden.py header).
  """
  mjm.vis.quality.offsamples = 0
  mjm.vis.headlight.active = 0


def _build_from_golden(data):
  from mjlab.entity import EntityCfg
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sensor import CameraSensorCfg
  from mjlab.sim.sim import SimulationCfg, make_simulation

  xml = str(data["xml"])
  cam_names = [str(n) for n in data["cam_names"]]
  height, width = int(data["height"]), int(data["width"])
  cam_cfgs = tuple(
    CameraSensorCfg(
      name=f"cam_{i}",
      camera_name=name,  # npz 存的是编译后模型的相机名
      width=width,
      height=height,
      data_types=("rgb", "depth"),
      render_backend="mjwarp",
    )
    for i, name in enumerate(cam_names)
  )
  entities = {"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))}
  scene = Scene(
    SceneCfg(
      num_envs=data["qpos"].shape[0],
      env_spacing=5.0,
      entities=entities,
      sensors=cam_cfgs,
    ),
    "cpu",
  )
  model = scene.compile()
  sim = make_simulation(
    num_envs=int(data["qpos"].shape[0]),
    cfg=SimulationCfg(backend="classic", njmax=20),
    model=model,
    device="cpu",
  )
  _apply_golden_visuals(sim.mj_model)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  sim.set_sensor_context(scene.sensor_context)

  # 固定状态集 → 派生场刷新（不 step，黄金帧按同状态渲染）。mjlab 编译可能
  # 追加 raw XML 没有的字段（如 mocap），形状一致才覆盖。
  for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat"):
    src = torch.from_numpy(data[name])
    dst = getattr(sim.data, name)
    if dst.shape == src.shape:
      dst[:] = src
  sim.forward()
  return scene, sim


def _assert_rgb_tolerance(got: np.ndarray, ref: np.ndarray, tag: str) -> None:
  """uint8 平均绝对差 ≤ 2 且 99% 像素（三通道最大差）≤ 4——写死，不放宽。"""
  assert got.shape == ref.shape and got.dtype == np.uint8
  diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
  mean_abs = float(diff.mean())
  per_pixel = diff.max(-1)
  p99 = float(np.percentile(per_pixel, 99))
  print(f"\n{tag}: rgb mean-abs {mean_abs:.3f}, p99 {p99:.1f}, "
        f"frac>4 {(per_pixel > 4).mean() * 100:.2f}%")
  assert mean_abs <= 2.0, f"{tag}: rgb 平均绝对差 {mean_abs:.3f} > 2"
  assert p99 <= 4.0, f"{tag}: rgb p99 像素差 {p99:.1f} > 4"


def _assert_depth_tolerance(
  got: np.ndarray, ref: np.ndarray, mjm: mujoco.MjModel, tag: str
) -> None:
  """命中像素 atol=1e-2/rtol=1e-2；背景集合一致（warp=0 ⟺ mjr=zfar 哨兵）。"""
  zfar = float(mjm.vis.map.zfar) * float(mjm.stat.extent)
  hit_ref = (ref > 0) & (ref < zfar * 0.999)
  hit_got = got > 0
  # 背景语义一致：无命中像素集合必须逐像素相同（真错位/裁剪差会被此抓出）。
  assert np.array_equal(hit_got, hit_ref), (
    f"{tag}: depth 背景集合不一致（命中像素数 got={int(hit_got.sum())} "
    f"ref={int(hit_ref.sum())}）"
  )
  err = np.abs(got - ref)[hit_ref]
  atol, rtol = 1e-2, 1e-2  # 上游 mujoco_warp render_test 同口径，写死
  worst = float((err - rtol * np.abs(ref[hit_ref])).max()) if err.size else 0.0
  assert worst <= atol, (
    f"{tag}: depth 最大超差 {max(err.max(), 0):.5f}（atol={atol}, rtol={rtol}）"
  )


@pytest.mark.parametrize("npz_path", GOLDEN_NPZ, ids=lambda p: p.stem)
def test_golden_rgb_parity(npz_path):
  data = np.load(npz_path)
  scene, sim = _build_from_golden(data)
  try:
    sim.sense()
    ncam = len(data["cam_names"])
    for ci in range(ncam):
      sensor = scene[f"cam_{ci}"]
      got = sensor.data.rgb.numpy()
      _assert_rgb_tolerance(got, data["rgb"][ci], f"{npz_path.stem}/cam{ci}")
  finally:
    sim.close()


@pytest.mark.parametrize("npz_path", GOLDEN_NPZ, ids=lambda p: p.stem)
def test_golden_depth_parity(npz_path):
  data = np.load(npz_path)
  scene, sim = _build_from_golden(data)
  try:
    sim.sense()
    mjm = sim.mj_model
    ncam = len(data["cam_names"])
    for ci in range(ncam):
      sensor = scene[f"cam_{ci}"]
      got = sensor.data.depth
      assert got is not None and got.shape == (sim.num_envs, int(data["height"]),
                                               int(data["width"]), 1)
      assert got.dtype == torch.float32
      _assert_depth_tolerance(
        got[..., 0].numpy(), data["depth"][ci], mjm, f"{npz_path.stem}/cam{ci}"
      )
  finally:
    sim.close()


# ---------------------------------------------------------------------------
# Step 2b: 形状 / dtype / 确定性 / 逐环境状态正确性。
# ---------------------------------------------------------------------------


def test_shapes_dtypes_and_determinism():
  scene, sim = _make(num_envs=4, data_types=("rgb", "depth"))
  try:
    sim.reset()
    sim.step()
    sim.sense()
    sensor = scene["test_cam"]
    rgb, depth = sensor.data.rgb, sensor.data.depth
    assert rgb.shape == (4, 84, 84, 3) and rgb.dtype == torch.uint8
    assert depth.shape == (4, 84, 84, 1) and depth.dtype == torch.float32
    assert int(rgb.float().sum()) > 0
    prev_rgb, prev_depth = rgb.clone(), depth.clone()
    sim.sense()  # 同状态重复渲染确定
    assert torch.equal(prev_rgb, sensor.data.rgb)
    assert torch.equal(prev_depth, sensor.data.depth)
  finally:
    sim.close()


def test_env_state_drives_frames():
  """逐环境 qpos 平移 → 逐环境图像不同（批内状态拷贝正确性旁证）。"""
  scene, sim = _make(num_envs=3)
  try:
    sim.reset()
    sim.data.qpos[:, 0] = torch.tensor([0.0, 0.2, 0.4])
    sim.forward()
    sim.sense()
    rgb = scene["test_cam"].data.rgb
    assert not torch.equal(rgb[0], rgb[1])
    assert not torch.equal(rgb[1], rgb[2])
  finally:
    sim.close()


# ---------------------------------------------------------------------------
# Step 4（先行断言）: 后端选择接线。
# ---------------------------------------------------------------------------


def test_backend_selection():
  from mjlab.sensor.camera_cpu import CpuCameraContext
  from mjlab.sensor.render_mjwarp import MjwarpCameraContext

  scene_w, sim_w = _make(num_envs=2, render_backend="mjwarp")
  try:
    assert type(sim_w and scene_w.sensor_context.camera_context) is MjwarpCameraContext
  finally:
    sim_w.close()

  scene_g, sim_g = _make(num_envs=2, render_backend="gl")
  try:
    assert type(scene_g.sensor_context.camera_context) is CpuCameraContext
  finally:
    sim_g.close()

  # auto = mjwarp 可导入则用之（mjlab 硬依赖 mujoco_warp，实际恒为 mjwarp）。
  scene_a, sim_a = _make(num_envs=2, render_backend="auto")
  try:
    assert type(scene_a.sensor_context.camera_context) is MjwarpCameraContext
  finally:
    sim_a.close()

  # 混合后端配置 → 明确报错。
  with pytest.raises(ValueError, match="render_backend"):
    _make_two_sensors_mixed()


def _make_two_sensors_mixed():
  from mjlab.entity import EntityCfg
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sensor import CameraSensorCfg
  from mjlab.sim.sim import SimulationCfg, make_simulation

  cfgs = (
    CameraSensorCfg(name="c1", camera_name="world/overhead_cam", width=32,
                    height=32, render_backend="gl"),
    CameraSensorCfg(name="c2", camera_name="world/overhead_cam", width=32,
                    height=32, render_backend="mjwarp"),
  )
  entities = {"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(SCENE_XML))}
  scene = Scene(SceneCfg(num_envs=1, env_spacing=5.0, entities=entities, sensors=cfgs), "cpu")
  model = scene.compile()
  sim = make_simulation(num_envs=1, cfg=SimulationCfg(backend="classic"),
                        model=model, device="cpu")
  scene.initialize(sim.mj_model, sim.model, sim.data)


def test_segmentation_unsupported():
  with pytest.raises(NotImplementedError, match="segmentation"):
    _make(num_envs=2, data_types=("rgb", "segmentation"))


# ---------------------------------------------------------------------------
# Step 2c: mjwarp 后端 + async_render 组合（正确性部分）。
# ---------------------------------------------------------------------------


def test_async_mjwarp_static_parity():
  """静止场景：mjwarp 后端 async 与 sync 像素逐位一致。"""
  scene_s, sim_s = _make(num_envs=4, async_render=False)
  scene_a, sim_a = _make(num_envs=4, async_render=True)
  try:
    for sim in (sim_s, sim_a):
      sim.reset()
      sim.data.qvel[:] = 0.0
      sim.forward()
    sim_s.sense()
    ref = scene_s["test_cam"].data.rgb.clone()
    sim_a.sense()  # 首帧阻塞等完成
    assert torch.equal(ref, scene_a["test_cam"].data.rgb.clone())
    sim_a.sense()
    assert torch.equal(ref, scene_a["test_cam"].data.rgb.clone())
  finally:
    sim_s.close()
    sim_a.close()


# ---------------------------------------------------------------------------
# Step 5: 性能门槛。
# ---------------------------------------------------------------------------

_NEEDS_CORES = pytest.mark.skipif(
  (os.cpu_count() or 1) < 8,
  reason="多进程分片渲染需要 >=8 核才有门槛意义",
)


def _measure_sense(sim, rounds: int, samples: int) -> float:
  """min-of-samples × rounds sense() 耗时（ms），并打印测量时负载。

  本仓库各 perf 门槛均为共享机器上的 wall-clock 口径：协作进程（同仓库其
  他 track 的测试）会造成 ±80% 抖动，多轮取 min 是稳定的成本估计。
  """
  best = float("inf")
  for _ in range(rounds):
    times = []
    for _ in range(samples):
      t0 = time.perf_counter()
      sim.sense()
      times.append((time.perf_counter() - t0) * 1e3)
    best = min(best, min(times))
    time.sleep(0.2)
  load = os.getloadavg()[0]
  print(f"(load avg {load:.1f} / {os.cpu_count()} cores)")
  return best


@_NEEDS_CORES
def test_perf_sync_mjwarp_64envs():
  """同步 mjwarp 渲染 ≤ 70ms @64envs×84×84（多进程分片，默认 worker 数）。"""
  scene, sim = _make(num_envs=64)
  try:
    sim.reset()
    sim.step()
    sim.sense()  # 首帧含 put_model/编译等一次性成本
    best = _measure_sense(sim, rounds=3, samples=5)
    print(f"sync mjwarp render @B=64,84x84: best {best:.1f} ms (gate 70ms)")
    assert best <= 70.0, f"mjwarp 渲染 {best:.1f} ms 超门槛 70ms"
  finally:
    sim.close()


@_NEEDS_CORES
def test_perf_async_mjwarp_sense_latency():
  """async 组合：sense() 调用 <1ms（30 次采样取 min，同 G-track 口径）。"""
  scene, sim = _make(num_envs=64, async_render=True)
  try:
    sim.reset()
    sim.forward()
    sim.sense()
    for _ in range(3):
      sim.sense()
    times = []
    for _ in range(30):
      t0 = time.perf_counter()
      sim.sense()
      times.append((time.perf_counter() - t0) * 1e3)
    best = min(times)
    print(f"\nasync sense() @B=64 mjwarp: min {best:.3f} ms (gate <1ms)")
    assert best < 1.0
  finally:
    sim.close()


@_NEEDS_CORES
def test_perf_async_mjwarp_throughput():
  """async 渲染吞吐 ≥ 同步 mjwarp 路径 80%（async min ≤ 1.25 × sync min）。

  sync/async 交错采样（3 轮），使两侧覆盖相同的机器负载窗口——共享机器上
  顺序测量会因负载突变产生假的吞吐比。
  """
  scene_s, sim_s = _make(num_envs=64, async_render=False)
  scene_a, sim_a = _make(num_envs=64, async_render=True)
  try:
    sim_s.reset()
    sim_s.step()
    sim_s.sense()
    sim_a.reset()
    sim_a.forward()
    sim_a.sense()
    ctx = scene_a.sensor_context.camera_context

    sync_best = async_best = float("inf")
    for _ in range(3):
      for _ in range(5):
        t0 = time.perf_counter()
        sim_s.sense()
        sync_best = min(sync_best, (time.perf_counter() - t0) * 1e3)
      for _ in range(3):
        k0 = ctx.latest_submitted_frame_id
        t0 = time.perf_counter()
        sim_a.sense()
        ctx.wait_for_frame(k0 + 1, timeout=60.0)
        async_best = min(async_best, (time.perf_counter() - t0) * 1e3)
      time.sleep(0.2)

    print(f"\nmjwarp throughput @B=64 (load {os.getloadavg()[0]:.1f}): "
          f"sync {sync_best:.1f} ms | async {async_best:.1f} ms | "
          f"ratio {async_best / sync_best:.2f}")
    assert async_best <= 1.25 * sync_best, (
      f"async {async_best:.1f} ms > 1.25 × sync {sync_best:.1f} ms（吞吐 <80%）"
    )
  finally:
    sim_s.close()
    sim_a.close()
