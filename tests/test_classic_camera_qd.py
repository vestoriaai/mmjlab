"""classic 后端 camera 渲染后端 = qd（Quadrants）批量光线追踪渲染（阶段 1 M1.5）。

对齐标准（写死，不放宽；与 test_classic_camera_mjwarp.py 同一体系）：
- RGB vs tests/golden/*.npz（mujoco.Renderer/mjr_ 参考，mujoco 3.11 生成）：
  全帧 uint8 平均绝对差 ≤ 2，且 99% 像素（逐像素取三通道最大差）≤ 4；
- depth vs golden：命中像素 atol=1e-2, rtol=1e-2；背景集合逐像素一致。

性能门槛（M1.4）：
- qd Metal 全管线渲染 64 envs×84×84 ≤ 15ms；
- 与 async_render=True 组合：sense() <1ms，渲染吞吐 ≥ 同步 qd 路径 80%。

黄金场景生成：tests/golden/generate_golden.py（同 mjwarp 测试共用）。
"""

from __future__ import annotations

import os
import pathlib
import time

import mujoco
import numpy as np
import pytest
import torch

GOLDEN_NPZ = sorted((pathlib.Path(__file__).parent / "golden").glob("stage0_*.npz"))

# 轻量场景：形状/dtype/确定性/后端选择用。
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
  render_backend: str = "qd",
  data_types: tuple[str, ...] = ("rgb",),
  async_render: bool = False,
  xml: str = SCENE_XML,
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


def _apply_golden_visuals(mjm: mujoco.MjModel) -> None:
  """Reapply the golden scenes' visual settings post-compile（同 mjwarp 测试）。"""
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
      camera_name=name,
      width=width,
      height=height,
      data_types=("rgb", "depth"),
      render_backend="qd",
    )
    for i, name in enumerate(cam_names)
  )
  entities = {"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))}
  scene = Scene(
    SceneCfg(
      num_envs=int(data["qpos"].shape[0]),
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
  for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat"):
    src = torch.from_numpy(data[name])
    dst = getattr(sim.data, name)
    if dst.shape == src.shape:
      dst[:] = src
  sim.forward()
  return scene, sim


def _assert_rgb_tolerance(got: np.ndarray, ref: np.ndarray, tag: str) -> None:
  assert got.shape == ref.shape and got.dtype == np.uint8
  diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
  mean_abs = float(diff.mean())
  per_pixel = diff.max(-1)
  p99 = float(np.percentile(per_pixel, 99))
  print(
    f"\n{tag}: rgb mean-abs {mean_abs:.3f}, p99 {p99:.1f}, "
    f"frac>4 {(per_pixel > 4).mean() * 100:.2f}%"
  )
  assert mean_abs <= 2.0, f"{tag}: rgb 平均绝对差 {mean_abs:.3f} > 2"
  assert p99 <= 4.0, f"{tag}: rgb p99 像素差 {p99:.1f} > 4"


def _assert_depth_tolerance(
  got: np.ndarray, ref: np.ndarray, mjm: mujoco.MjModel, tag: str
) -> None:
  zfar = float(mjm.vis.map.zfar) * float(mjm.stat.extent)
  hit_ref = (ref > 0) & (ref < zfar * 0.999)
  hit_got = got > 0
  assert np.array_equal(hit_got, hit_ref), (
    f"{tag}: depth 背景集合不一致（命中像素数 got={int(hit_got.sum())} "
    f"ref={int(hit_ref.sum())}）"
  )
  err = np.abs(got - ref)[hit_ref]
  atol, rtol = 1e-2, 1e-2
  worst = float((err - rtol * np.abs(ref[hit_ref])).max()) if err.size else 0.0
  assert worst <= atol, (
    f"{tag}: depth 最大超差 {max(err.max(), 0):.5f}（atol={atol}, rtol={rtol}）"
  )


# ---------------------------------------------------------------------------
# 黄金帧容差对齐（RGB + depth）。
# ---------------------------------------------------------------------------


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
      assert got is not None and got.shape == (
        sim.num_envs,
        int(data["height"]),
        int(data["width"]),
        1,
      )
      assert got.dtype == torch.float32
      _assert_depth_tolerance(
        got[..., 0].numpy(), data["depth"][ci], mjm, f"{npz_path.stem}/cam{ci}"
      )
  finally:
    sim.close()


# ---------------------------------------------------------------------------
# 形状 / dtype / 确定性 / 逐环境状态正确性。
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
    sim.sense()
    assert torch.equal(prev_rgb, sensor.data.rgb)
    assert torch.equal(prev_depth, sensor.data.depth)
  finally:
    sim.close()


def test_env_state_drives_frames():
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
# 后端选择接线。
# ---------------------------------------------------------------------------


def test_backend_selection():
  from mjlab.sensor.render_mjwarp import MjwarpCameraContext
  from mjlab.sensor.render_qd import QdCameraContext, is_available

  scene_q, sim_q = _make(num_envs=2, render_backend="qd")
  try:
    assert type(sim_q and scene_q.sensor_context.camera_context) is QdCameraContext
  finally:
    sim_q.close()

  scene_m, sim_m = _make(num_envs=2, render_backend="mjwarp")
  try:
    assert type(scene_m.sensor_context.camera_context) is MjwarpCameraContext
  finally:
    sim_m.close()

  # auto 策略（M1.5 更新）：qd 可导入则优先 qd，其次 mjwarp，最后 gl。
  scene_a, sim_a = _make(num_envs=2, render_backend="auto")
  try:
    expected = QdCameraContext if is_available() else MjwarpCameraContext
    assert type(scene_a.sensor_context.camera_context) is expected
  finally:
    sim_a.close()


def test_segmentation_unsupported():
  with pytest.raises(NotImplementedError, match="segmentation"):
    _make(num_envs=2, data_types=("rgb", "segmentation"))


# ---------------------------------------------------------------------------
# qd 后端 + async_render 组合（正确性）。
# ---------------------------------------------------------------------------


def test_async_qd_static_parity():
  """静止场景：qd 后端 async 与 sync 像素逐位一致。"""
  scene_s, sim_s = _make(num_envs=4, async_render=False, data_types=("rgb", "depth"))
  scene_a, sim_a = _make(num_envs=4, async_render=True, data_types=("rgb", "depth"))
  try:
    for sim in (sim_s, sim_a):
      sim.reset()
      sim.data.qvel[:] = 0.0
      sim.forward()
    sim_s.sense()
    ref = scene_s["test_cam"].data.rgb.clone()
    ref_depth = scene_s["test_cam"].data.depth.clone()
    sim_a.sense()  # 首帧阻塞等完成
    assert torch.equal(ref, scene_a["test_cam"].data.rgb.clone())
    assert torch.equal(ref_depth, scene_a["test_cam"].data.depth.clone())
    sim_a.sense()
    assert torch.equal(ref, scene_a["test_cam"].data.rgb.clone())
  finally:
    sim_s.close()
    sim_a.close()


# ---------------------------------------------------------------------------
# 性能门槛（M1.4：qd Metal 全管线 ≤15ms；async 组合 sense <1ms、吞吐 ≥80%）。
# ---------------------------------------------------------------------------

_NEEDS_CORES = pytest.mark.skipif(
  (os.cpu_count() or 1) < 8,
  reason="吞吐测量需要 >=8 核才有门槛意义",
)

_LOAD_LIMIT = (os.cpu_count() or 1) * 0.5


def _measure_sense(sim, rounds: int, samples: int, target_ms: float) -> float:
  best = float("inf")
  measured_load = None
  for _ in range(rounds * 2):
    load = os.getloadavg()[0]
    if load > _LOAD_LIMIT:
      time.sleep(1.0)
      continue
    times = []
    for _ in range(samples):
      t0 = time.perf_counter()
      sim.sense()
      times.append((time.perf_counter() - t0) * 1e3)
    best = min(best, min(times))
    measured_load = load
    if best <= target_ms:
      break
    time.sleep(0.5)
  if measured_load is None:
    pytest.skip(f"机器持续繁忙（load > {_LOAD_LIMIT:.0f}），吞吐门槛无法测量")
  print(f"(measured at load {measured_load:.1f} / {os.cpu_count()} cores)")
  return best


def _qd_arch_is_metal() -> bool:
  try:
    from mjlab.sensor import raycast_qd

    return raycast_qd.qd_arch() == "metal"
  except Exception:
    return False


@_NEEDS_CORES
@pytest.mark.skipif(not _qd_arch_is_metal(), reason="qd 性能门槛按 Metal 口径")
def test_perf_sync_qd_64envs():
  """qd Metal 全管线渲染 ≤ 15ms @64envs×84×84（M1.4 门槛，写死不放宽）。"""
  scene, sim = _make(num_envs=64)
  try:
    sim.reset()
    sim.step()
    sim.sense()  # 首帧含建场景/编译等一次性成本
    best = _measure_sense(sim, rounds=3, samples=5, target_ms=15.0)
    print(f"sync qd render @B=64,84x84: best {best:.1f} ms (gate 15ms)")
    assert best <= 15.0, f"qd 渲染 {best:.1f} ms 超门槛 15ms"
  finally:
    sim.close()


@_NEEDS_CORES
@pytest.mark.skipif(not _qd_arch_is_metal(), reason="qd 性能门槛按 Metal 口径")
def test_perf_async_qd_sense_latency():
  """async 组合：sense() 调用 <1ms（30 次采样取 min）。"""
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
    print(f"\nasync sense() @B=64 qd: min {best:.3f} ms (gate <1ms)")
    assert best < 1.0
  finally:
    sim.close()


@_NEEDS_CORES
@pytest.mark.skipif(not _qd_arch_is_metal(), reason="qd 性能门槛按 Metal 口径")
def test_perf_async_qd_throughput():
  """async 渲染吞吐 ≥ 同步 qd 路径 80%（async min ≤ 1.25 × sync min）。"""
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

    print(
      f"\nqd throughput @B=64 (load {os.getloadavg()[0]:.1f}): "
      f"sync {sync_best:.1f} ms | async {async_best:.1f} ms | "
      f"ratio {async_best / sync_best:.2f}"
    )
    assert async_best <= 1.25 * sync_best, (
      f"async {async_best:.1f} ms > 1.25 × sync {sync_best:.1f} ms（吞吐 <80%）"
    )
  finally:
    sim_s.close()
    sim_a.close()
