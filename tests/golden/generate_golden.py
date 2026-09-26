"""Generate golden reference frames for classic-camera renderer parity (stage 0).

Renders fixed state sets of three test scenes (primitives / mesh / hfield) with
``mujoco.Renderer`` (the mjr_ reference) and stores everything needed to
rebuild and compare against them in ``tests/golden/*.npz``:

  xml          scene MJCF (str)
  cam_names    MuJoCo camera names used as sensors (ncam,)
  width/height image resolution (scalars; one resolution per scene)
  qpos/qvel/act/mocap_pos/mocap_quat  per-env states (nenv, ...)
  rgb          reference RGB  (ncam, nenv, H, W, 3) uint8
  depth        reference depth (ncam, nenv, H, W) float32, metric planar depth
               (mjr enable_depth_rendering; background pixels are 0)

The scenes are aligned for tight cross-renderer tolerance (see
docs/results/stage0-mjwarp-render.md): anti-aliasing off (offsamples=0, same
convention as upstream mujoco_warp render_test mjr comparisons), headlight
off, a single no-specular directional light. Reference frames are rendered
from the raw state vectors (mj_kinematics + mj_comPos + mj_camlight in a
scratch MjData), independent of any mjlab pipeline.

Usage: python tests/golden/generate_golden.py
"""

from __future__ import annotations

import os
import pathlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
NENV_DYNAMIC = 4  # scenes with a movable body: env w shifts its x by w*X_SHIFT
X_SHIFT = 0.12

_HEAD = '<visual><quality offsamples="0"/><headlight active="0"/></visual>'
_LIGHT = '<light pos="0 0 3" dir="0 0 -1" directional="true" specular="0 0 0"/>'

# Camera slightly tilted down; single box keeps silhouette-edge divergence
# (GL rasterizer vs per-pixel rays, inherent) well inside the tolerance budget.
SCENES = {
  "stage0_primitives": f"""
<mujoco>{_HEAD}
  <worldbody>
    {_LIGHT}
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
    <camera name="top_cam" pos="0 -2.0 2.2" xyaxes="1 0 0 0 0.55 0.84"
            fovy="45" resolution="84 84"/>
    <camera name="side_cam" pos="-1.6 -0.6 0.9" xyaxes="0.94 0.34 0 0 0 1"
            fovy="45" resolution="84 84"/>
    <body name="floater" pos="0 0 0.3">
      <freejoint name="floater_joint"/>
      <geom name="box" type="box" size="0.15 0.15 0.15" rgba="1 0 0 1" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
""",
  "stage0_mesh": f"""
<mujoco>{_HEAD}
  <asset>
    <mesh name="cube" vertex="-0.15 -0.15 -0.15  -0.15 -0.15 0.15  -0.15 0.15 -0.15  -0.15 0.15 0.15
                              0.15 -0.15 -0.15  0.15 -0.15 0.15  0.15 0.15 -0.15  0.15 0.15 0.15"/>
  </asset>
  <worldbody>
    {_LIGHT}
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
    <camera name="cam" pos="0 -2.0 2.2" xyaxes="1 0 0 0 0.55 0.84"
            fovy="45" resolution="84 84"/>
    <body name="floater" pos="0 0 0.3">
      <freejoint name="floater_joint"/>
      <geom name="m" type="mesh" mesh="cube" rgba="0.9 0.6 0.1 1" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
""",
  # Gentle terrain: hfield sampling/shading matches mjr within <=1 uint8 level
  # when AA/headlight/lighting are pinned (see worklog-S0 spike 4/5).
  "stage0_hfield": f"""
<mujoco>{_HEAD}
  <asset>
    <hfield name="hf" nrow="5" ncol="5" size="1.2 1.2 0.12 0.05"
            elevation="0 0 1 0 0
                       0 1 2 1 0
                       1 2 3 2 1
                       0 1 2 1 0
                       0 0 1 0 0"/>
  </asset>
  <worldbody>
    {_LIGHT}
    <camera name="cam" pos="0 -2.2 1.8" xyaxes="1 0 0 0 0.45 0.89"
            fovy="45" resolution="84 84"/>
    <geom name="terrain" type="hfield" hfield="hf" rgba="0.3 0.7 0.3 1"/>
  </worldbody>
</mujoco>
""",
}


def scene_states(mjm: mujoco.MjModel) -> dict[str, np.ndarray]:
  """Fixed per-env state vectors: env w shifts the free body x by w*X_SHIFT."""
  nenv = NENV_DYNAMIC if mjm.nq else 1
  out = {}
  for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat"):
    out[name] = np.stack(
      [np.array(getattr(mujoco.MjData(mjm), name)).copy() for _ in range(nenv)]
    )
  for w in range(nenv):
    if mjm.nq:
      out["qpos"][w, 0] = w * X_SHIFT
  return out


def render_reference(mjm: mujoco.MjModel, states: dict[str, np.ndarray]):
  """mjr reference: per env RGB and metric planar depth for every camera."""
  nenv = states["qpos"].shape[0]
  height = int(mjm.cam_resolution[0, 1])
  width = int(mjm.cam_resolution[0, 0])
  cam_names = [mjm.camera(i).name for i in range(mjm.ncam)]
  rgb = np.zeros((len(cam_names), nenv, height, width, 3), dtype=np.uint8)
  depth = np.zeros((len(cam_names), nenv, height, width), dtype=np.float32)

  renderer = mujoco.Renderer(mjm, height=height, width=width)
  scratch = mujoco.MjData(mjm)
  try:
    for w in range(nenv):
      scratch.qpos[:] = states["qpos"][w]
      scratch.qvel[:] = states["qvel"][w]
      scratch.act[:] = states["act"][w]
      scratch.mocap_pos[:] = states["mocap_pos"][w]
      scratch.mocap_quat[:] = states["mocap_quat"][w]
      mujoco.mj_kinematics(mjm, scratch)
      mujoco.mj_comPos(mjm, scratch)
      mujoco.mj_camlight(mjm, scratch)
      for ci, cam_id in enumerate(range(mjm.ncam)):
        renderer.update_scene(scratch, camera=cam_id)
        rgb[ci, w] = renderer.render()
        renderer.enable_depth_rendering()
        depth[ci, w] = renderer.render()
        renderer.disable_depth_rendering()
  finally:
    renderer.close()
  return cam_names, rgb, depth


def main() -> None:
  for tag, xml in SCENES.items():
    mjm = mujoco.MjModel.from_xml_string(xml)
    states = scene_states(mjm)
    cam_names, rgb, depth = render_reference(mjm, states)
    path = HERE / f"{tag}.npz"
    np.savez_compressed(
      path,
      xml=xml,
      cam_names=np.array(cam_names),
      width=int(mjm.cam_resolution[0, 0]),
      height=int(mjm.cam_resolution[0, 1]),
      rgb=rgb,
      depth=depth,
      **states,
    )
    print(
      f"{path.name}: {len(cam_names)} cams x {rgb.shape[1]} envs @"
      f"{rgb.shape[3]}x{rgb.shape[2]}, non-black px "
      f"{(rgb.sum(-1) > 0).mean() * 100:.1f}%, depth range "
      f"[{depth[depth > 0].min():.2f}, {depth[depth > 0].max():.2f}]"
    )


if __name__ == "__main__":
  main()
