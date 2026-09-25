"""classic 后端 diff-IK：批量 torch 雅可比 vs mujoco.mj_jacBody 逐环境 f64 参考。

对齐容差（固定）：rel err < 1e-5。性能门槛：_jac_torch ≤ 0.15ms @B=256。
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import mujoco
import numpy as np
import torch

NUM_ENVS = 4


def _make_classic_arm():
  """classic 后端 + 带位置执行器的三连杆臂（复用 diff-IK 官方测试的 ARM_XML 结构）。"""
  from mjlab.actuator.actuator import TransmissionType
  from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
  from mjlab.sim.sim import MujocoCfg, SimulationCfg, make_simulation

  xml = """\
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.05 0.02" mass="1.0"
            contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.3"
              size="0.02" mass="0.5" contype="0" conaffinity="0"/>
        <body name="link2" pos="0 0 0.3">
          <joint name="joint2" type="slide" axis="0 0 1" range="-0.2 0.2"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.3"
                size="0.02" mass="0.5" contype="0" conaffinity="0"/>
          <body name="ee" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 1 0"
                   range="-3.14 3.14"/>
            <geom name="ee_geom" type="sphere" size="0.03" mass="0.1"
                  contype="0" conaffinity="0"/>
            <site name="ee_site" pos="0 0 0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint.*",),
          transmission_type=TransmissionType.JOINT,
          stiffness=10.0,
          damping=1.0,
          effort_limit=10.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = make_simulation(
    num_envs=NUM_ENVS,
    cfg=SimulationCfg(backend="classic", mujoco=MujocoCfg(gravity=(0, 0, 0))),
    model=model,
    device="cpu",
  )
  entity.initialize(model, sim.model, sim.data, "cpu")
  return entity, sim


def _reference_jacs(model: mujoco.MjModel, sim, body_id: int, point: np.ndarray):
  """逐环境 f64 参考：mj_kinematics + mj_comPos + mj_jac（任意参考点）。"""
  B = sim.num_envs
  nv = model.nv
  jacp_ref = np.zeros((B, 3, nv))
  jacr_ref = np.zeros((B, 3, nv))
  scratch = mujoco.MjData(model)
  qpos = sim.data.qpos.detach().cpu().numpy()
  qvel = sim.data.qvel.detach().cpu().numpy()
  for w in range(B):
    scratch.qpos[:] = qpos[w]
    scratch.qvel[:] = qvel[w]
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    jp = np.zeros((3, nv))
    jr = np.zeros((3, nv))
    mujoco.mj_jac(model, scratch, jp, jr, point[w], body_id)
    jacp_ref[w] = jp
    jacr_ref[w] = jr
  return jacp_ref, jacr_ref


def test_jacobian_matches_mj_jac_body():
  from mjlab.envs.mdp.actions.differential_ik import _jac_torch

  entity, sim = _make_classic_arm()
  model = sim.mj_model
  sim.reset()
  # 非平凡状态
  rng = np.random.default_rng(0)
  qvel = sim.data.qvel.detach().cpu().numpy()
  qvel[:] = rng.uniform(-0.5, 0.5, qvel.shape)
  sim.step()

  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee")
  cdof = sim.data.cdof.detach().cpu().numpy().astype(np.float64)
  subtree_com = sim.data.subtree_com.detach().cpu().numpy().astype(np.float64)
  # 非平凡参考点（目标 body 所在 weld 根的 subtree com，即动作项运行时的参考点）
  point = subtree_com[:, model.body_rootid[body_id], :]
  jacp_ref, jacr_ref = _reference_jacs(model, sim, body_id, point)
  jacp, jacr = _jac_torch(
    torch.tensor(cdof),
    torch.tensor(subtree_com),
    model.dof_bodyid,
    model.body_rootid,
    model.body_parentid,
    torch.tensor(point),
    body_id,
  )
  for got, ref, name in ((jacp, jacp_ref, "jacp"), (jacr, jacr_ref, "jacr")):
    got = got.detach().cpu().numpy().astype(np.float64)
    denom = np.abs(ref).max() + 1e-12
    rel = np.abs(got - ref) / denom
    assert rel.max() < 1e-5, f"{name} rel err {rel.max():.2e}"


def test_classic_diff_ik_action_runs():
  """e2e：classic 后端上 DifferentialIKAction 处理/应用动作 10 步保持有限值。"""
  from mjlab.envs.mdp.actions import DifferentialIKActionCfg
  from mjlab.envs.mdp.actions.differential_ik import DifferentialIKAction

  entity, sim = _make_classic_arm()
  env = Mock(spec=["num_envs", "device", "scene", "sim"])
  env.num_envs = NUM_ENVS
  env.device = "cpu"
  env.scene = {"robot": entity}
  env.sim = sim

  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    use_relative_mode=True,
  )
  action: DifferentialIKAction = cfg.build(env)
  assert action.action_dim == 6

  rng = np.random.default_rng(1)
  for _ in range(10):
    actions = torch.tensor(
      rng.uniform(-0.05, 0.05, (NUM_ENVS, action.action_dim)),
      dtype=torch.float32,
    )
    action.process_actions(actions)
    dq = action.compute_dq()
    assert torch.isfinite(dq).all()
    action.apply_actions()
    sim.step()
  assert np.isfinite(np.asarray(sim.data.qpos)).all()


def test_jac_torch_perf_256():
  """性能门槛：_jac_torch ≤ 0.15ms @B=256（humanoid 模型）。"""
  import os

  from mjlab.envs.mdp.actions.differential_ik import _jac_torch

  humanoid_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "mujoco_warp",
    "benchmarks",
    "humanoid",
    "humanoid.xml",
  )
  print("loading humanoid...", flush=True)
  model = mujoco.MjModel.from_xml_path(humanoid_path)
  nv = model.nv
  print("nv =", nv, flush=True)
  B = 256
  rng = np.random.default_rng(2)
  cdof = torch.tensor(rng.standard_normal((B, nv, 6)))
  subtree_com = torch.tensor(rng.standard_normal((B, model.nbody, 3)))
  point = subtree_com[:, 0, :]
  body_id = 1

  print("inputs ready", flush=True)

  def run():
    _jac_torch(
      cdof,
      subtree_com,
      model.dof_bodyid,
      model.body_rootid,
      model.body_weldid,
      point,
      body_id,
    )

  for _ in range(20):
    run()
  times = []
  for _ in range(50):
    t0 = time.perf_counter()
    run()
    times.append((time.perf_counter() - t0) * 1e3)
  best = min(times)
  print(
    f"\n_jac_torch @B=256,nv={nv}: best {best:.4f} ms (median {np.median(times):.4f})"
  )
  assert best <= 0.15, f"_jac_torch {best:.4f} ms 超预算 0.15ms @nv={nv}"
