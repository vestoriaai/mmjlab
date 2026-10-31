"""流水线边际成本:阶段级 + kernel 级 GPU 成本归因(qd.metal 大 batch 下 GPU-bound).

方法:某阶段/kernel 单独连续执行 R 次、中间不 sync,吞吐 = 流水线内真实成本
(GPU-bound 时为 GPU 时间;GPU 时间小于 host 发起开销时测到的是 host 底噪)。

- 阶段级:forward() 的各阶段函数 + 积分器,逐个重复执行。solve 每次重新
  init,迭代数与真实一致。
- kernel 级:录下一步内全部 kernel 调用(对象 + 参数),逐个重放。solver
  内部 kernel 跳过(步末 ctx.done 全真,重放会早退,低估),由阶段级 solve 覆盖。

Usage: python scripts/stage_cost_qd.py --num-envs 4096
"""

from __future__ import annotations

import argparse
import collections
import importlib
import time

import mujoco

_DEFAULT_XML = "/Users/mini0/Code/simtest/mujoco_warp/benchmarks/humanoid/humanoid.xml"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--device", default="mps")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--reps", type=int, default=40)
  p.add_argument("--task", default=None, help="真实任务名(不填则用 humanoid bare)")
  p.add_argument("--njmax", type=int, default=None)
  args = p.parse_args()

  if args.task:
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
    act = torch.zeros(env.num_envs, sum(env.action_manager.action_term_dim), device=env.device)
    for _ in range(40):
      env.step(act)
    sim = env.sim
  else:
    from mjlab.sim.sim import SimulationCfg, make_simulation

    model = mujoco.MjModel.from_xml_path(_DEFAULT_XML)
    sim = make_simulation(num_envs=args.num_envs, cfg=SimulationCfg(backend="qd"),
                          model=model, device=args.device)
    sim.reset()
  import quadrants as qd
  import mujoco_qd as mjw
  from mujoco_qd import IntegratorType
  from mujoco_qd._src import forward as F
  from mujoco_qd._src import sensor, smooth, solver

  qk = importlib.import_module("quadrants.lang.kernel")
  m, d = sim.wp_model, sim.wp_data
  for _ in range(200):
    mjw.step(m, d)
  qd.sync()

  R = args.reps

  def cost(fn):
    fn()
    qd.sync()
    t0 = time.perf_counter()
    for _ in range(R):
      fn()
    qd.sync()
    return (time.perf_counter() - t0) / R

  integ = {
    IntegratorType.EULER: F.euler,
    IntegratorType.IMPLICITFAST: F.implicit,
    IntegratorType.IMPLICIT: F.implicit,
  }.get(m.opt.integrator)

  def sensor_acc():
    sensor.sensor_acc(m, d, skip_rne_postconstraint=True)

  stages = [
    ("fwd_position", lambda: F.fwd_position(m, d, factorize=False)),
    ("sensor_pos", lambda: sensor.sensor_pos(m, d)),
    ("energy_pos", lambda: F._energy_pos(m, d)),
    ("fwd_velocity", lambda: F.fwd_velocity(m, d)),
    ("sensor_vel", lambda: sensor.sensor_vel(m, d)),
    ("energy_vel", lambda: F._energy_vel(m, d)),
    ("fwd_actuation", lambda: F.fwd_actuation(m, d)),
    ("fwd_acceleration", lambda: F.fwd_acceleration(m, d, factorize=True)),
    ("solver.solve", lambda: solver.solve(m, d)),
    ("rne_postconstraint", lambda: smooth.rne_postconstraint(m, d)),
    ("sensor_acc", sensor_acc),
    (f"integrator({IntegratorType(m.opt.integrator).name})", lambda: integ(m, d)),
  ]
  t_step = cost(lambda: mjw.step(m, d))
  print(f"envs={args.num_envs} nv={m.nv} iterations={m.opt.iterations} "
        f"integrator={IntegratorType(m.opt.integrator).name} solver_niter(mean)="
        f"{d.solver_niter.to_numpy().mean():.2f}")
  print(f"\n[A] 阶段级流水线成本(整步 {t_step * 1e3:.3f} ms)")
  rows = []
  for name, fn in stages:
    try:
      rows.append((name, cost(fn)))
    except Exception as e:  # 某些阶段在该模型下不适用
      print(f"  {name}: skipped ({type(e).__name__}: {e})")
  tot = sum(c for _, c in rows)
  for name, c in sorted(rows, key=lambda r: -r[1]):
    print(f"  {name:<28s} {c * 1e3:8.3f} ms  {c / tot * 100:5.1f}%")
  print(f"  {'合计':<28s} {tot * 1e3:8.3f} ms  (整步 {t_step * 1e3:.3f} ms)")

  # [B] kernel 级:录一步内的调用序列
  calls = []
  orig = qk.Kernel.__call__

  def rec(self, *a, **k):
    calls.append((self, a, dict(k)))
    return orig(self, *a, **k)

  qk.Kernel.__call__ = rec
  mjw.step(m, d)
  qd.sync()
  qk.Kernel.__call__ = orig

  per_name: collections.Counter = collections.Counter()
  cnt: collections.Counter = collections.Counter()
  skipped = 0
  for kobj, a, k in calls:
    fn = getattr(kobj, "func", None)
    mod = getattr(fn, "__module__", "?").rsplit(".", 1)[-1]
    name = f"{mod}.{getattr(fn, '__name__', '?')}"
    if mod in ("solver", "block_cholesky_coop", "support"):
      skipped += 1
      continue
    per_name[name] += cost(lambda: orig(kobj, *a, **k))
    cnt[name] += 1
  floor = min(per_name.values()) if per_name else 0.0
  print(f"\n[B] kernel 级流水线成本(非 solver,{len(calls) - skipped} 次调用;"
        f"最小单次 {floor * 1e6:.1f} µs ≈ host 发起底噪)")
  tot_b = sum(per_name.values())
  for name, c in per_name.most_common(30):
    print(f"  {name:<56s} {c * 1e3:7.3f} ms  {cnt[name]:3d}x  {c / cnt[name] * 1e6:8.1f} µs/call")
  print(f"  {'合计(非 solver kernel)':<56s} {tot_b * 1e3:7.3f} ms")


if __name__ == "__main__":
  main()
