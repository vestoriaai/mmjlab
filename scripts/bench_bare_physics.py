"""Bare-physics throughput sanity: qd vs classic vs upstream warp on CPU.

Usage (inside the venv that has the mjlab flavour under test):
  python scripts/bench_bare_physics.py --backend qd     --num-envs 256 [--xml PATH]
  python scripts/bench_bare_physics.py --backend classic --num-envs 256
  python scripts/bench_bare_physics.py --backend warp    --num-envs 256  # upstream mjlab venv

Protocol: fixed XML, zero ctrl, min-of-R repeats, warmup excluded. This is the
engine-only signal (no observations/rewards/sensors); the training-throughput
matrix is a separate, later measurement.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import torch

_DEFAULT_XML = "/Users/mini0/Code/simtest/mujoco_warp/benchmarks/humanoid/humanoid.xml"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=["qd", "classic", "warp"], default="qd")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--xml", type=str, default=_DEFAULT_XML)
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(args.xml)
  backend = "qd" if args.backend == "warp" else args.backend  # upstream: warp only
  from mjlab.sim.sim import SimulationCfg, make_simulation

  sim = make_simulation(
    num_envs=args.num_envs,
    cfg=SimulationCfg(backend=backend),
    model=model,
    device="cpu",
  )
  sim.reset()

  # Warmup (kernel compilation / allocator warmup excluded from timing).
  for _ in range(20):
    sim.step()

  best = float("inf")
  for _ in range(args.repeats):
    t0 = time.perf_counter()
    for _ in range(args.steps):
      sim.step()
    best = min(best, time.perf_counter() - t0)

  sps = args.num_envs * args.steps / best
  print(
    f"backend={args.backend} envs={args.num_envs} xml={args.xml.rsplit('/', 1)[-1]}: "
    f"{sps:,.0f} steps/s (min of {args.repeats} x {args.steps} steps, "
    f"{best * 1000:.1f} ms/wall-step)"
  )


if __name__ == "__main__":
  main()
