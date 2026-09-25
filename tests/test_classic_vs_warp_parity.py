"""classic(f64 C) vs mjwarp(f32) 单步一致性。容差反映 f32 引擎差异，非 bug。

API 调用形式以 Task A4 Step 2 实测记录为准（yuvaltassa#1 rebase 后的分支）:
    from mjlab.sim.sim import SimulationCfg, make_simulation
    sim = make_simulation(num_envs=N, cfg=SimulationCfg(backend=...),
                          model=mujoco.MjModel, device="cpu")   # 全部 kw-only
    sim.reset() / sim.step() / sim.forward() / sim.data.qpos -> torch f32 (nworld, nq)
计划初稿的 Simulation(cfg, model, nworld) 签名不存在，已按上述真实签名修正夹具；
数值断言 1e-3 未放宽。"""

import mujoco
import numpy as np
import pytest

from cartpole_xml import CARTPOLE_XML


def _make(backend: str, nworld: int):
    from mjlab.sim.sim import SimulationCfg, make_simulation

    model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    cfg = SimulationCfg(backend=backend)
    return make_simulation(num_envs=nworld, cfg=cfg, model=model, device="cpu")


@pytest.mark.parametrize("backend", ["classic", "warp"])
def test_one_step_finite(backend):
    sim = _make(backend, nworld=4)
    sim.reset()
    sim.step()
    qpos = sim.data.qpos
    assert qpos.shape[0] == 4 and np.isfinite(np.asarray(qpos)).all()


def test_classic_vs_warp_one_step_close():
    qs = {}
    for backend in ("classic", "warp"):
        sim = _make(backend, nworld=4)
        sim.reset()
        np.random.default_rng(0)
        sim.data.qvel[:] = 0.05
        sim.step()
        qs[backend] = np.asarray(sim.data.qpos)
    diff = np.abs(qs["classic"] - qs["warp"]).max()
    assert diff < 1e-3, f"单步 qpos 最大偏差 {diff}（f64 vs f32 容差 1e-3 rad）"
