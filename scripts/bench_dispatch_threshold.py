"""Dispatch micro-bench: per-launch cost vs task count (serial threshold 64).

Answers: what does ntasks<=64 inline vs the pool wake cost look like on this
machine, and how does per-launch overhead scale.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/Users/mini0/Code/simtest/mmjlab")

import numpy as np
import quadrants as qd
from mujoco_qd._src.warp_util import from_data

qd.init(arch=qd.cpu, cfg_optimization=False)


@qd.kernel
def spin(arr: qd.types.ndarray(dtype=float, ndim=1), out: qd.types.ndarray(dtype=float, ndim=1)):
  for t in range(arr.shape[0]):
    s = 0.0
    for k in range(200):  # ~fixed per-task work
      s += arr[t] * k
    out[t] = s


for n in (16, 63, 64, 65, 256, 4096, 73728):
  a = from_data(np.zeros(n))
  o = from_data(np.zeros(n))
  spin(arr=a, out=o)  # compile + warm
  t0 = time.perf_counter()
  for _ in range(20):
    spin(arr=a, out=o)
  dt = (time.perf_counter() - t0) / 20 * 1e3
  mode = "INLINE" if n <= 64 else "pool"
  print(f"[tasks={n:>6}] {dt:9.3f} ms/launch  ({mode})", flush=True)
