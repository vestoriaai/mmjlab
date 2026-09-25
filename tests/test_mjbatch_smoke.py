import mujoco
import numpy as np
from mjbatch import Batch

from cartpole_xml import CARTPOLE_XML

NSIMS = 64
NSTEPS = 100


def test_batch_bit_identical_to_mj_step_loop():
    model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    batch = Batch(model, num_sims=NSIMS)
    qpos, qvel = batch.bind("qpos"), batch.bind("qvel")
    qpos[:] = 0.01
    qvel[:] = 0.0

    data = [mujoco.MjData(model) for _ in range(NSIMS)]
    for d in data:
        d.qpos[:] = 0.01

    for _ in range(NSTEPS):
        batch.step()
        for d in data:
            mujoco.mj_step(model, d)

    ref = np.stack([d.qpos for d in data])
    got = np.asarray(qpos)
    assert np.array_equal(got, ref), f"max diff {np.abs(got - ref).max()}"
