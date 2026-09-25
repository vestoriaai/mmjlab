"""classic 后端 camera 传感器：mujoco.Renderer 逐环境渲染。

对齐要求：管线输出与独立 mujoco.Renderer 参考渲染逐像素相等。
性能门槛：rgb ≤ 200ms @B=64, 84×84。
"""

from __future__ import annotations

import time

import mujoco
import pytest
import torch

# floater：freejoint 动体，验证逐环境状态拷贝正确性（静态场景图像不随 qvel 变）。
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


def _make(num_envs: int = 4, height: int = 84, width: int = 84):
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


def test_camera_shapes_and_determinism():
  scene, sim = _make(num_envs=4, height=84, width=84)
  sim.reset()
  sim.step()
  sim.sense()
  sensor = scene["test_cam"]
  rgb = sensor.data
  assert rgb.rgb is not None
  assert rgb.rgb.shape == (4, 84, 84, 3)
  assert rgb.rgb.dtype == torch.uint8
  # 非全黑（场景有 light + 红色 box + 地板）
  assert int(rgb.rgb.float().sum()) > 0
  # 同状态重复渲染确定
  sim.sense()
  assert torch.equal(rgb.rgb, scene["test_cam"].data.rgb)
  # 不同环境状态 → 图像不同（state copy 正确性旁证）：平移 floater 0.3m。
  # 注：rgb 视图与 warp 语义一致，sense 会就地覆盖，先 clone 旧图。
  prev = rgb.rgb.clone()
  sim.data.qpos[:, 0] += 0.3
  sim.forward()
  sim.sense()
  assert not torch.equal(prev, scene["test_cam"].data.rgb)


def test_camera_matches_standalone_renderer():
  """管线输出与独立 mujoco.Renderer 参考渲染逐像素相等。"""
  scene, sim = _make(num_envs=4, height=84, width=84)
  sim.reset()
  sim.data.qvel[:] = 0.1
  sim.step()
  sim.sense()
  pipeline = scene["test_cam"].data.rgb

  model = sim.mj_model
  cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "world/overhead_cam")
  reference = mujoco.Renderer(model, height=84, width=84)
  scratch = mujoco.MjData(model)
  try:
    for w in range(sim.num_envs):
      scratch.qpos[:] = sim.data.qpos[w].numpy()
      scratch.qvel[:] = sim.data.qvel[w].numpy()
      scratch.act[:] = sim.data.act[w].numpy()
      scratch.mocap_pos[:] = sim.data.mocap_pos[w].numpy()
      scratch.mocap_quat[:] = sim.data.mocap_quat[w].numpy()
      mujoco.mj_kinematics(model, scratch)
      mujoco.mj_comPos(model, scratch)
      mujoco.mj_camlight(model, scratch)
      reference.update_scene(scratch, camera=cam_id)
      ref = reference.render()
      assert torch.equal(pipeline[w], torch.from_numpy(ref)), f"env {w} 逐像素不等"
  finally:
    reference.close()


def test_camera_rejects_depth_and_segmentation():
  from mjlab.entity import EntityCfg
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sensor import CameraSensorCfg
  from mjlab.sim.sim import SimulationCfg, make_simulation

  cam_cfg = CameraSensorCfg(
    name="cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb", "depth"),
  )
  entities = {"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(SCENE_XML))}
  scene = Scene(
    SceneCfg(num_envs=2, env_spacing=5.0, entities=entities, sensors=(cam_cfg,)),
    "cpu",
  )
  model = scene.compile()
  sim = make_simulation(
    num_envs=2, cfg=SimulationCfg(backend="classic"), model=model, device="cpu"
  )
  with pytest.raises(NotImplementedError, match="depth"):
    scene.initialize(sim.mj_model, sim.model, sim.data)


def test_camera_perf_64():
  """性能门槛：rgb 渲染 ≤ 200ms @B=64, 84×84。"""
  scene, sim = _make(num_envs=64, height=84, width=84)
  sim.reset()
  sim.step()
  sim.sense()
  times = []
  for _ in range(5):
    t0 = time.perf_counter()
    sim.sense()
    times.append((time.perf_counter() - t0) * 1e3)
  best = min(times)
  print(f"\nclassic camera rgb @B=64,84x84: best {best:.1f} ms (budget 200ms)")
  assert best <= 200, f"camera sense {best:.1f} ms 超预算 200ms"
