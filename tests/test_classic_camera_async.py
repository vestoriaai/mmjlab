"""classic 后端 camera 异步渲染（CameraSensorCfg.async_render=True）。

正确性要求：
- 静止场景 async=False 与 async=True 像素逐位一致（同状态渲染确定性）；
- 运动场景（匀速平移）async 帧对应状态滞后 ≤1 步（∈ {当前, 上一步}）；
- async 连跑 300 步无死锁/无异常；close() 后再 sense() 明确报错。

性能门槛（统计口径 = 全采样最小值 min，理由：共享机器桌面负载使中位数波动
±15% 以上，min 是无噪声的成本估计；当前记录时段负载 5.5-6.6，见 worklog-G）：
- async 开启时 sense() 调用本身 <1ms（渲染在后台线程）；
- 渲染线程吞吐 ≥ 同步实现的 80%：64 envs×84×84 每帧 ≤130ms，
  且同轮实测 async min ≤ 1.25 × sync min（≥80% 吞吐的相对口径）。
"""

from __future__ import annotations

import time

import mujoco
import pytest
import torch

from mjlab.sensor.camera_cpu import AsyncCpuCameraContext, CpuCameraContext

# floater：freejoint 动体（同 test_classic_camera 场景），供运动滞后测试平移。
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


def _make(num_envs: int, height: int = 84, width: int = 84, async_render: bool = False):
  from mjlab.entity import EntityCfg
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sensor import CameraSensorCfg
  from mjlab.sim.sim import SimulationCfg, make_simulation

  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=width,
    height=height,
    data_types=("rgb",),
    async_render=async_render,
  )
  entities = {"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(SCENE_XML))}
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


def _red_centroid(rgb: torch.Tensor) -> torch.Tensor:
  """逐 env 红色 box 像素质心 (row, col)，shape [B, 2]。

  同一状态的渲染逐位一致 → 质心可作位置指纹（容差仅浮点误差级）。
  """
  r = rgb[..., 0].int()
  g = rgb[..., 1].int()
  b = rgb[..., 2].int()
  mask = (r >= 150) & (g <= 90) & (b <= 90)
  num_envs = rgb.shape[0]
  centroids = torch.zeros(num_envs, 2, dtype=torch.float64)
  for w in range(num_envs):
    rows, cols = torch.nonzero(mask[w], as_tuple=True)
    if rows.numel() == 0:
      raise AssertionError(f"env {w} 未找到红色 box 像素（掩码为空）")
    centroids[w, 0] = rows.double().mean()
    centroids[w, 1] = cols.double().mean()
  return centroids


def _set_state(sim, k: int, delta: float = 0.09) -> None:
  """把场景设为第 k 拍的确定状态：floater x = k*delta（运动学平移）。"""
  sim.reset()
  sim.data.qvel[:] = 0.0
  sim.data.qpos[:, 0] = k * delta
  sim.forward()


def test_async_default_off_and_ctx_selection():
  """默认 async_render=False（零变化合同）；ctx 类型随开关正确选择。"""
  from mjlab.sensor import CameraSensorCfg

  assert CameraSensorCfg(
    name="c", camera_name="x", width=8, height=8
  ).async_render is False

  scene_s, sim_s = _make(num_envs=2, async_render=False)
  try:
    assert type(scene_s.sensor_context.camera_context) is CpuCameraContext
    assert not hasattr(scene_s.sensor_context.camera_context, "_thread")
  finally:
    sim_s.close()

  scene_a, sim_a = _make(num_envs=2, async_render=True)
  try:
    ctx = scene_a.sensor_context.camera_context
    assert type(ctx) is AsyncCpuCameraContext
    assert ctx._thread.is_alive()
  finally:
    sim_a.close()


def test_async_static_pixels_bitwise_match():
  """静止场景：async=False vs async=True 像素逐位一致；二次 sense 仍一致。"""
  scene_s, sim_s = _make(num_envs=4, async_render=False)
  scene_a, sim_a = _make(num_envs=4, async_render=True)
  try:
    for sim in (sim_s, sim_a):
      sim.reset()
      sim.data.qvel[:] = 0.0
      sim.forward()

    sim_s.sense()
    ref = scene_s["test_cam"].data.rgb.clone()
    assert int(ref.float().sum()) > 0

    t0 = time.perf_counter()
    sim_a.sense()  # 首次调用：无已完成帧 → 等待首帧完成（同步语义兜底）
    first_ms = (time.perf_counter() - t0) * 1e3
    got = scene_a["test_cam"].data.rgb.clone()
    print(f"\nasync first sense (first-frame wait): {first_ms:.1f} ms")
    assert got.shape == ref.shape
    assert torch.equal(ref, got), "静止场景 async 与 sync 像素不等"

    sim_a.sense()
    assert torch.equal(scene_a["test_cam"].data.rgb, ref), "同状态二次渲染不一致"
  finally:
    sim_s.close()
    sim_a.close()


def test_async_frame_lag_at_most_one_step():
  """运动场景：async 帧里物体位置 ∈ {当前步, 上一步}。

  floater 逐拍 +0.09m（≈3-4px/拍，质心可分辨）；参考质心由 sync 渲染同一组
  状态预先固化，async 每拍质心必须恰等于参考[当前] 或参考[上一步]。
  节拍说明：async 语义为"队列只保留最新任务、立即返回最近完成帧"——渲染
  持续跟不上 sense 频率时完成帧可落后更多（v1 合同，主线程不阻塞）。本测
  试在"渲染跟得上"的现实节拍下验证 ≤1 步：每拍间隔 100ms，远大于 2 envs
  单帧渲染耗时（~5-8ms），读取帧只能来自当前或上一步状态。
  """
  k_max = 8
  scene_s, sim_s = _make(num_envs=2, async_render=False)
  try:
    refs = []
    for k in range(k_max + 1):
      _set_state(sim_s, k)
      sim_s.sense()
      refs.append(_red_centroid(scene_s["test_cam"].data.rgb.clone()))
    # 相邻拍参考质心必须可分辨（≥1.5px），否则断言无区分度。
    for k in range(1, k_max + 1):
      step_px = float(torch.max(torch.abs(refs[k] - refs[k - 1])))
      assert step_px >= 1.5, f"参考质心步进 {step_px:.2f}px 过小（k={k}）"
  finally:
    sim_s.close()

  scene_a, sim_a = _make(num_envs=2, async_render=True)
  try:
    sim_a.reset()
    for k in range(k_max + 1):
      _set_state(sim_a, k)
      sim_a.sense()
      got = _red_centroid(scene_a["test_cam"].data.rgb.clone())
      allowed = [refs[k]] + ([refs[k - 1]] if k >= 1 else [])
      dists = [float(torch.max(torch.abs(got - r))) for r in allowed]
      assert min(dists) < 1e-6, (
        f"step {k}: 质心 {got[0].tolist()} 不在允许集 "
        f"{[r[0].tolist() for r in allowed]}（滞后 >1 步或状态错位）"
      )
      time.sleep(0.1)  # 节拍控制：给渲染线程留足完成窗口（见 docstring）
  finally:
    sim_a.close()


def test_async_stability_300_steps_and_close():
  """async 开启连跑 300 步无死锁/无异常；close() 幂等且 sense() 明确报错。"""
  scene_a, sim_a = _make(num_envs=4, async_render=True)
  ctx = scene_a.sensor_context.camera_context
  try:
    sim_a.reset()
    for k in range(300):
      sim_a.data.qpos[:, 0] += 0.001  # 300 步累计 0.3m，保持视野内
      sim_a.forward()
      sim_a.sense()
      if k % 50 == 0:
        rgb = scene_a["test_cam"].data.rgb
        assert rgb is not None and int(rgb.float().sum()) > 0
    assert ctx.latest_completed_frame_id is not None
  finally:
    sim_a.close()

  assert not ctx._thread.is_alive(), "close() 后渲染线程未退出"
  ctx.close()  # 幂等
  with pytest.raises(RuntimeError, match="closed"):
    sim_a.sense()
  with pytest.raises(RuntimeError, match="closed"):
    scene_a["test_cam"].data.rgb  # noqa: B018


def test_async_sense_call_latency():
  """性能：async 开启时 sense() 调用本身 <1ms（渲染在后台）。

  统计口径：30 次采样取 min（共享机器 ±15% 抖动，见 docstring）。
  """
  scene_a, sim_a = _make(num_envs=64, async_render=True)
  try:
    sim_a.reset()
    sim_a.forward()
    sim_a.sense()  # 首帧阻塞（建立已完成帧槽位）
    for _ in range(3):
      sim_a.sense()
    times = []
    for _ in range(30):
      t0 = time.perf_counter()
      sim_a.sense()
      times.append((time.perf_counter() - t0) * 1e3)
    best = min(times)
    median = sorted(times)[len(times) // 2]
    print(
      f"\nasync sense() latency @B=64: min {best:.3f} ms, median {median:.3f} ms "
      f"(budget <1ms)"
    )
    assert best < 1.0, f"async sense() min {best:.3f} ms 超预算 1ms"
  finally:
    sim_a.close()


def test_async_render_thread_throughput():
  """性能：渲染线程吞吐 ≥ 同步 80%（每帧 ≤130ms，且 ≤1.25×sync min）。

  统计口径：两侧均为同轮实测取 min；async 测法 = 提交后 wait_for_frame
  该帧完成（串行测得线程单帧净耗时）。
  """
  # sync 基线（同场景同分辨率，协议同 test_camera_perf_64）。
  scene_s, sim_s = _make(num_envs=64, async_render=False)
  try:
    sim_s.reset()
    sim_s.step()
    sim_s.sense()
    sync_times = []
    for _ in range(5):
      t0 = time.perf_counter()
      sim_s.sense()
      sync_times.append((time.perf_counter() - t0) * 1e3)
    sync_best = min(sync_times)
  finally:
    sim_s.close()

  scene_a, sim_a = _make(num_envs=64, async_render=True)
  try:
    sim_a.reset()
    sim_a.forward()
    sim_a.sense()  # 首帧
    ctx = scene_a.sensor_context.camera_context
    async_times = []
    for _ in range(7):
      k0 = ctx.latest_submitted_frame_id
      t0 = time.perf_counter()
      sim_a.sense()  # 提交第 k0+1 帧
      ctx.wait_for_frame(k0 + 1, timeout=60.0)
      async_times.append((time.perf_counter() - t0) * 1e3)
    async_best = min(async_times)
  finally:
    sim_a.close()

  print(
    f"\nrender throughput @B=64,84x84: sync min {sync_best:.1f} ms | "
    f"async min {async_best:.1f} ms | ratio {async_best / sync_best:.2f} "
    f"(gates: async <=130ms, ratio <=1.25)"
  )
  assert async_best <= 130.0, f"async 每帧 {async_best:.1f} ms 超门槛 130ms"
  assert async_best <= 1.25 * sync_best, (
    f"async 每帧 {async_best:.1f} ms > 1.25 × sync min {sync_best:.1f} ms "
    f"（吞吐 <80%）"
  )