"""classic 后端对 sleep 配置的优雅兼容。

事实依据（F3 Step 1 探测，见 docs/results/worklog-F.md）：
mujoco 3.11 的 C 引擎存在 `mjENBL_SLEEP`，但启用后接触场景入睡即产生
~1e-6 级轨迹分叉（第 192 步实测 9.1e-7）→ 违反「sleep-on 与 sleep-off
bit-identical」合同 → classic 剥离该 flag 并 warning，原 cfg 对象不被就地修改。
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from cartpole_xml import CARTPOLE_XML

FALLING_BOXES_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="b1" pos="0 0 0.2"><freejoint/>
      <geom name="g1" type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
    <body name="b2" pos="0.3 0 0.2"><freejoint/>
      <geom name="g2" type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
    <body name="b3" pos="0.6 0 0.2"><freejoint/>
      <geom name="g3" type="box" size="0.05 0.05 0.05" mass="0.1"/></body>
  </worldbody>
</mujoco>
"""


def _make(xml: str, flags, num_envs: int = 4):
  from mjlab.sim.sim import MujocoCfg, SimulationCfg, make_simulation

  model = mujoco.MjModel.from_xml_string(xml)
  cfg = SimulationCfg(backend="classic", mujoco=MujocoCfg(enableflags=tuple(flags)))
  return cfg, make_simulation(num_envs=num_envs, cfg=cfg, model=model, device="cpu")


def test_classic_accepts_sleep_flag():
  """enableflags=("sleep",) 可构建、可步进：flag 被剥离 + warning，不落到 mj_model。"""
  with pytest.warns(UserWarning, match="sleep"):
    cfg, sim = _make(CARTPOLE_XML, ("sleep",))
  sim.reset()
  sim.step()
  assert np.isfinite(np.asarray(sim.data.qpos)).all()
  sleep_bit = int(mujoco.mjtEnableBit.mjENBL_SLEEP)
  assert sim.mj_model.opt.enableflags & sleep_bit == 0
  # 原 cfg 对象未被就地修改
  assert cfg.mujoco.enableflags == ("sleep",)


def test_sleep_on_equals_off_bitwise():
  """sleep 配置与不配置的 classic 轨迹 bit-identical（含接触入睡场景）。"""
  sims = {}
  for name, flags, warns in (("on", ("sleep",), True), ("off", (), False)):
    if warns:
      with pytest.warns(UserWarning, match="sleep"):
        _, sims[name] = _make(FALLING_BOXES_XML, flags)
    else:
      _, sims[name] = _make(FALLING_BOXES_XML, flags)
    sims[name].reset()
    sims[name].data.qvel[:] = 0.05
  for step in range(50):
    sims["on"].step()
    sims["off"].step()
    assert np.array_equal(
      np.asarray(sims["on"].data.qpos), np.asarray(sims["off"].data.qpos)
    ), f"step {step} 后轨迹分叉"
  assert torch.isfinite(sims["on"].data.qpos).all()


def test_other_enable_flags_still_apply():
  """非 sleep 的 enable flag 照常透传（energy）。"""
  _, sim = _make(CARTPOLE_XML, ("energy",))
  assert sim.mj_model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_ENERGY)
