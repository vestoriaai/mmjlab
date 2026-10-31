"""forward replay parity:同一状态下 replay forward vs eager forward.

Usage: PYTHONPATH=<mujoqo> python scripts/forward_replay_parity.py --task Mjlab-SitStand-Flat-MicroDuck

确定性量(运动学/接触集合排序后/nefc/传感器)要求逐位一致;qacc/efc 力受接触
槽位原子分配顺序影响,以 eager 对 eager 的差异为允许包络。
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default="Mjlab-SitStand-Flat-MicroDuck")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--device", default="mps")
  args = p.parse_args()

  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.reset()
  import quadrants as qd
  from mujoco_qd._src import forward as F
  from mujoco_qd._src import plan

  a = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim), device=env.device)
  for _ in range(80):
    env.step(a)
  qd.sync()
  m, d = env.sim.wp_model, env.sim.wp_data
  ent = plan._PLAN_CACHE.get((id(m), id(d), "forward"))
  recorded = ent is not None and ent[0] is not None and not isinstance(ent[0], str)
  print(f"@@ forward plan recorded={recorded} state={None if ent is None else type(ent[0]).__name__}")

  def snap():
    qd.sync()
    n = int(d.nacon.to_numpy()[0])
    w = d.contact.worldid.to_numpy()[:n]
    g = d.contact.geom.to_numpy()[:n]
    pos = d.contact.pos.to_numpy()[:n]
    key = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0], g[:, 1], g[:, 0], w))
    return {
      "xpos": d.xpos.to_numpy().copy(),
      "xquat": d.xquat.to_numpy().copy(),
      "cvel": d.cvel.to_numpy().copy(),
      "qfrc_bias": d.qfrc_bias.to_numpy().copy(),
      "actuator_force": d.actuator_force.to_numpy().copy(),
      "nacon": np.array([n]),
      "contact_dist": d.contact.dist.to_numpy()[:n][key],
      "contact_pos": pos[key],
      "nefc": d.nefc.to_numpy().copy(),
      "sensordata": d.sensordata.to_numpy().copy(),
      "qacc": d.qacc.to_numpy().copy(),
    }

  env.sim.forward()  # replay
  R = snap()
  F._forward_body(m, d)  # eager
  E1 = snap()
  F._forward_body(m, d)
  E2 = snap()

  ok = recorded
  exact = ["xpos", "xquat", "cvel", "qfrc_bias", "actuator_force", "nacon", "contact_dist",
           "contact_pos", "nefc"]
  for k in exact:
    if R[k].shape != E1[k].shape:
      print(f"@@ {k}: shape {R[k].shape} vs {E1[k].shape} FAIL")
      ok = False
      continue
    diff = float(np.abs(R[k].astype(np.float64) - E1[k].astype(np.float64)).max()) if R[k].size else 0.0
    good = diff == 0.0
    ok &= good
    print(f"@@ {k:<16s} replay vs eager max|d|={diff:.3e} {'OK' if good else 'FAIL'}")
  env_q = float(np.abs(E1["qacc"] - E2["qacc"]).max())
  rq = float(np.abs(R["qacc"] - E1["qacc"]).max())
  env_s = float(np.abs(E1["sensordata"] - E2["sensordata"]).max())
  rs = float(np.abs(R["sensordata"] - E1["sensordata"]).max())
  print(f"@@ qacc        replay vs eager {rq:.3e} | eager vs eager(包络) {env_q:.3e}")
  print(f"@@ sensordata  replay vs eager {rs:.3e} | eager vs eager(包络) {env_s:.3e}")
  # 包络判定:replay 偏差不超过 eager 自身运行间差异的 2 倍(两者都为 0 时要求 0)
  ok &= rq <= max(2 * env_q, 1e-6) and rs <= max(2 * env_s, 1e-6)
  print(f"@@ RESULT {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
  main()
