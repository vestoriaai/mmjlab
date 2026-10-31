"""qd 后端单步 profile:封装层开销 / host 底噪 / 阶段拆分 / kernel 调用计数.

Usage:
  python scripts/profile_qd_step.py --device mps --num-envs 4096
  python scripts/profile_qd_step.py --device mps --num-envs 64   # host 底噪

输出:
  1. sim.step()(含 nan_guard)vs mujoco_qd.step() 直调的每步墙钟
  2. 不 sync 时 host 发起一步的耗时(host-bound 时 ≈ 墙钟)
  3. 每步 kernel 调用次数,按模块 / kernel 名聚合
  4. EventTracer 阶段树(每个 event_scope 结束时 qd.sync,阶段间串行化,
     适合归因,绝对值会比流水线执行偏大)
"""

from __future__ import annotations

import argparse
import collections
import time

import mujoco

_DEFAULT_XML = "/Users/mini0/Code/simtest/mujoco_warp/benchmarks/humanoid/humanoid.xml"


def _per_step(fn, n, sync):
  sync()
  t0 = time.perf_counter()
  for _ in range(n):
    fn()
  sync()
  return (time.perf_counter() - t0) / n


def _print_trace(node, depth=0, nsteps=1):
  for name, (events, sub) in sorted(node.items(), key=lambda kv: -sum(kv[1][0])):
    total_ms = sum(events) * 1e3 / nsteps
    calls = len(events) / nsteps
    print(f"  {'  ' * depth}{name:<40s} {total_ms:8.3f} ms/step  ({calls:.1f} calls/step)")
    if sub:
      _print_trace(sub, depth + 1, nsteps)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--device", default="mps")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--warmup", type=int, default=150)
  p.add_argument("--xml", default=_DEFAULT_XML)
  args = p.parse_args()

  from mjlab.sim.sim import SimulationCfg, make_simulation

  model = mujoco.MjModel.from_xml_path(args.xml)
  sim = make_simulation(num_envs=args.num_envs, cfg=SimulationCfg(backend="qd"),
                        model=model, device=args.device)
  sim.reset()

  import quadrants as qd
  import mujoco_qd as mjw
  from mujoco_qd._src import warp_util
  import importlib

  # quadrants.lang.kernel 属性被同名装饰器函数遮蔽,按模块路径取
  qk = importlib.import_module("quadrants.lang.kernel")

  m, d = sim.wp_model, sim.wp_data
  print(f"mujoco_qd: {mjw.__file__}")
  print(f"envs={args.num_envs} device={args.device} nv={m.nv} "
        f"iterations={m.opt.iterations} njmax={d.njmax}")

  for _ in range(args.warmup):
    sim.step()
  qd.sync()

  n = args.steps
  t_sim = _per_step(sim.step, n, qd.sync)
  t_core = _per_step(lambda: mjw.step(m, d), n, qd.sync)
  # host 发起时间:不 sync 的循环(GPU 跟不上时会被 command queue 背压拖慢)
  qd.sync()
  t0 = time.perf_counter()
  for _ in range(n):
    mjw.step(m, d)
  t_issue = (time.perf_counter() - t0) / n
  qd.sync()

  print("\n[1] 每步墙钟")
  print(f"  sim.step()(mmjlab 封装)  {t_sim * 1e3:8.3f} ms")
  print(f"  mujoco_qd.step() 直调      {t_core * 1e3:8.3f} ms")
  print(f"  封装层开销                 {(t_sim - t_core) * 1e3:8.3f} ms")
  print(f"  host 发起(不 sync)         {t_issue * 1e3:8.3f} ms")
  print(f"  吞吐(直调)                 {args.num_envs / t_core:,.0f} steps/s")

  by_name: collections.Counter = collections.Counter()
  by_mod: collections.Counter = collections.Counter()
  orig = qk.Kernel.__call__

  def counted(self, *a, **k):
    fn = getattr(self, "func", None)
    name = getattr(fn, "__name__", "?")
    mod = getattr(fn, "__module__", "?").rsplit(".", 1)[-1]
    by_name[f"{mod}.{name}"] += 1
    by_mod[mod] += 1
    return orig(self, *a, **k)

  qk.Kernel.__call__ = counted
  ncount = 20
  for _ in range(ncount):
    mjw.step(m, d)
  qd.sync()
  qk.Kernel.__call__ = orig
  total = sum(by_mod.values()) / ncount
  print(f"\n[2] kernel 调用:{total:.1f} 次/步(平均每次 {t_core * 1e6 / total:.1f} µs 墙钟)")
  for mod, c in by_mod.most_common():
    print(f"  {mod:<28s} {c / ncount:7.1f} /step")
  print("  -- top 25 kernels --")
  for name, c in by_name.most_common(25):
    print(f"  {name:<60s} {c / ncount:6.1f} /step")

  ntrace = 30
  with warp_util.EventTracer() as tr:
    for _ in range(ntrace):
      mjw.step(m, d)
    trace = tr.trace()
  print(f"\n[3] EventTracer 阶段树(阶段边界 sync,{ntrace} 步平均)")
  _print_trace(trace, 0, ntrace)

  # [4] 逐 kernel 串行计时:每次调用前后 sync,得到 launch+执行+sync 的完整耗时。
  # 与小 batch 的同表相减即得随 nworld 增长的 GPU 计算量。
  t_name: collections.Counter = collections.Counter()
  c_name: collections.Counter = collections.Counter()

  def timed_call(self, *a, **k):
    fn = getattr(self, "func", None)
    name = f"{getattr(fn, '__module__', '?').rsplit('.', 1)[-1]}.{getattr(fn, '__name__', '?')}"
    qd.sync()
    t0 = time.perf_counter()
    r = orig(self, *a, **k)
    qd.sync()
    t_name[name] += time.perf_counter() - t0
    c_name[name] += 1
    return r

  qk.Kernel.__call__ = timed_call
  nk = 20
  t0 = time.perf_counter()
  for _ in range(nk):
    mjw.step(m, d)
  qd.sync()
  t_serial = (time.perf_counter() - t0) / nk
  qk.Kernel.__call__ = orig
  t_kern = sum(t_name.values()) / nk
  print(f"\n[4] 逐 kernel 串行计时:串行化后每步 {t_serial * 1e3:.3f} ms,"
        f"其中 kernel 内 {t_kern * 1e3:.3f} ms,kernel 外 Python {(t_serial - t_kern) * 1e3:.3f} ms")
  for name, t in t_name.most_common(30):
    print(f"  {name:<60s} {t * 1e3 / nk:8.3f} ms/step  {c_name[name] / nk:5.1f}x  "
          f"{t * 1e6 / c_name[name]:8.1f} µs/call")


if __name__ == "__main__":
  main()
