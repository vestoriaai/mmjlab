"""真实任务吞吐 + QD_PLAN replay 覆盖率检查(qd 后端).

Usage: python scripts/env_throughput_qd.py --task Mjlab-Velocity-Flat-Unitree-Go1 --num-envs 256

输出 env.step / 纯物理(sim.step × decimation)吞吐,以及 plan 缓存状态:
replay 是否生效、eager 回退步数(planned_step 只在非 replay 路径上累加计数)。
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default="Mjlab-Velocity-Flat-Unitree-Go1")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--device", default="mps")
  p.add_argument("--warmup", type=int, default=60)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--njmax", type=int, default=None, help="覆盖 cfg.sim.njmax(实验用)")
  args = p.parse_args()

  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  if args.njmax is not None:
    env_cfg.sim.njmax = args.njmax
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env.reset()

  import quadrants as qd
  from mujoco_qd._src import plan

  action = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim), device=env.device)
  for _ in range(args.warmup):
    env.step(action)
  qd.sync()

  def plan_state():
    out = []
    for key, ent in plan._PLAN_CACHE.items():
      state = ent[0]
      kind = "replay" if (state is not None and not isinstance(state, str)) else (
        "impossible" if isinstance(state, str) else "not-recorded")
      out.append((key[2] if len(key) > 2 else "step", kind, ent[1]))
    return out

  before = plan_state()
  dec = env.cfg.decimation
  t0 = time.perf_counter()
  for _ in range(args.steps * dec):
    env.sim.step()
  qd.sync()
  phys = args.num_envs * args.steps / (time.perf_counter() - t0)

  t0 = time.perf_counter()
  for _ in range(args.steps):
    env.step(action)
  qd.sync()
  envsps = args.num_envs * args.steps / (time.perf_counter() - t0)

  nf = 30
  t0 = time.perf_counter()
  for _ in range(nf):
    env.sim.forward()
  qd.sync()
  t_fwd = (time.perf_counter() - t0) / nf
  after = plan_state()

  d = env.sim.wp_data
  import numpy as _np
  from mujoco_qd._src.types import OverflowType
  ovf = int(d.overflow.to_numpy().max()) if hasattr(d, "overflow") else -1
  ovf_names = [ot.name for ot in OverflowType if ovf & int(ot)] or ["none"]
  print(f"task={args.task} envs={args.num_envs} njmax={d.njmax} decimation={dec} plan.enabled={plan.enabled()} "
        f"nefc_max={int(d.nefc.to_numpy().max())} overflow={ovf} ({','.join(ovf_names)})")
  print(f"  physics: {phys:,.0f} env-steps/s  ({1e3 * args.num_envs / phys / dec:.3f} ms/sim.step)")
  print(f"  env.step: {envsps:,.0f} env-steps/s  (manager 开销 {100 * (1 - envsps / phys):.1f}%)")
  print(f"  sim.forward()(eager,不走 replay): {t_fwd * 1e3:.3f} ms/次")
  print(f"  plan 状态(测量前): {before}")
  print(f"  plan 状态(测量后): {after}   # (kind, 状态, 累计 eager 次数)")
  env.close()


if __name__ == "__main__":
  main()
