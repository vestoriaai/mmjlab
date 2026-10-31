"""cProfile 一步 mujoco_qd.step 的 host 路径,定位每次 launch 的 host 开销去向.

Usage: python scripts/cprofile_qd_step.py --device mps --num-envs 4096
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats

import mujoco

_DEFAULT_XML = "/Users/mini0/Code/simtest/mujoco_warp/benchmarks/humanoid/humanoid.xml"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--device", default="mps")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--steps", type=int, default=50)
  p.add_argument("--top", type=int, default=40)
  args = p.parse_args()

  from mjlab.sim.sim import SimulationCfg, make_simulation

  model = mujoco.MjModel.from_xml_path(_DEFAULT_XML)
  sim = make_simulation(num_envs=args.num_envs, cfg=SimulationCfg(backend="qd"),
                        model=model, device=args.device)
  sim.reset()
  import quadrants as qd
  import mujoco_qd as mjw

  m, d = sim.wp_model, sim.wp_data
  for _ in range(150):
    mjw.step(m, d)
  qd.sync()

  pr = cProfile.Profile()
  pr.enable()
  for _ in range(args.steps):
    mjw.step(m, d)
  qd.sync()
  pr.disable()

  for key in ("tottime", "cumulative"):
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats(key)
    st.print_stats(args.top)
    print(f"===== sort by {key} (per {args.steps} steps) =====")
    print("\n".join(s.getvalue().splitlines()[:args.top + 12]))


if __name__ == "__main__":
  main()
