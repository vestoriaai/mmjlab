"""Bare-physics profile: classic(mjbatch) / qd.cpu / qd.metal / upstream warp.

Usage (inside the venv that has the mjlab flavour under test):
  python scripts/bench_bare_physics.py --backend classic --num-envs 4096
  python scripts/bench_bare_physics.py --backend qd --device cpu  --num-envs 4096
  python scripts/bench_bare_physics.py --backend qd --device mps  --num-envs 4096
  python scripts/bench_bare_physics.py --backend warp --num-envs 256  # upstream mjlab venv

Protocol: fixed XML, zero ctrl, min-of-R repeats. Adaptive warmup: steps are
timed individually until the per-step time stabilizes (median-of-50 windows
within 5%), so engine JIT/compilation — minutes-long on a cold qd.metal run —
is fully excluded from the timed region and reported separately.

Memory: macOS ``ru_maxrss`` (process lifetime peak, bytes on macOS; includes
Metal/unified-memory device buffers) — absolute peak plus the peak delta
around simulation construction+warmup+run, and current RSS at run time.
"""

from __future__ import annotations

import argparse
import os
import resource
import statistics
import subprocess
import time

import mujoco

_DEFAULT_XML = "/Users/mini0/Code/simtest/mujoco_warp/benchmarks/humanoid/humanoid.xml"

_WARM_MIN_STEPS = 60
_WARM_MAX_STEPS = 4000


def _peak_rss_mb() -> float:
  """Lifetime peak RSS of this process (macOS ru_maxrss is in bytes)."""
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _current_rss_mb() -> float:
  """Current resident set via ps (KB on macOS)."""
  out = subprocess.run(
    ["ps", "-o", "rss=", "-p", str(os.getpid())],
    capture_output=True, text=True,
  ).stdout.strip()
  return float(out) / 1024.0


def _adaptive_warmup(sim) -> tuple[int, float, float]:
  """Step until per-step time stabilizes; returns (steps, seconds, first_step_s)."""
  times: list[float] = []
  t0 = time.perf_counter()
  first = None
  while len(times) < _WARM_MAX_STEPS:
    s = time.perf_counter()
    sim.step()
    dt = time.perf_counter() - s
    if first is None:
      first = dt
    times.append(dt)
    if len(times) >= 2 * 50 and len(times) >= _WARM_MIN_STEPS:
      recent = statistics.median(times[-50:])
      prev = statistics.median(times[-100:-50])
      if prev > 0 and abs(recent - prev) <= 0.05 * prev:
        break
  return len(times), time.perf_counter() - t0, first


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=["qd", "classic", "warp"], default="qd")
  parser.add_argument("--device", default="cpu", help="cpu | mps (qd.metal)")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--xml", type=str, default=_DEFAULT_XML)
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(args.xml)
  backend = "qd" if args.backend == "warp" else args.backend  # upstream: warp only
  from mjlab.sim.sim import Simulation, SimulationCfg

  try:
    from mjlab.sim.sim import make_simulation

    sim = make_simulation(
      num_envs=args.num_envs,
      cfg=SimulationCfg(backend=backend),
      model=model,
      device=args.device,
    )
  except ImportError:
    # Upstream mjlab (warp-only): construct the Simulation directly.
    sim = Simulation(num_envs=args.num_envs, cfg=SimulationCfg(), model=model,
                     device=args.device)
  sim.reset()

  warm_steps, warm_s, first_step_s = _adaptive_warmup(sim)

  rss_at_run = _current_rss_mb()
  peak_before_run = _peak_rss_mb()

  best = float("inf")
  for _ in range(args.repeats):
    t0 = time.perf_counter()
    for _ in range(args.steps):
      sim.step()
    best = min(best, time.perf_counter() - t0)

  sps = args.num_envs * args.steps / best
  peak_after_run = _peak_rss_mb()
  print(
    f"backend={args.backend} device={args.device} envs={args.num_envs} "
    f"xml={args.xml.rsplit('/', 1)[-1]}: {sps:,.0f} steps/s "
    f"(min of {args.repeats} x {args.steps} steps, {best * 1000:.1f} ms/wall-step) | "
    f"warmup {warm_steps} steps / {warm_s:.1f} s (first step {first_step_s * 1000:.0f} ms) | "
    f"rss@run {rss_at_run:.0f} MB | peak {peak_after_run:.0f} MB "
    f"(peak-delta {peak_after_run - peak_before_run:.0f} MB)"
  )


if __name__ == "__main__":
  main()
