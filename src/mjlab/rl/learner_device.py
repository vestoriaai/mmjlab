"""Experimental split-device PPO training (Apple Silicon).

Keeps the env, rollout storage, and policy inference on the runner device (CPU)
while migrating only the PPO learner — actor/critic modules plus the minibatch
stream — to a faster torch device (e.g. MPS) for the duration of each
``alg.update()`` call.  Rollout-facing state stays on the runner device, so the
per-step inference path pays no device-transfer tax; the only cross-device
traffic is module weights (~MB) and minibatches, both per-iteration.

Enable via MJLAB_LEARNER_DEVICE=<device> in scripts/train.py.
"""

import os

import torch
from tensordict import TensorDictBase


def _move(value, device: str):
  """Move tensors / TensorDicts / (nested) tensor containers to `device`."""
  if value is None or isinstance(value, (int, float, str, bool)):
    return value
  if isinstance(value, torch.Tensor):
    return value.to(device)
  if isinstance(value, TensorDictBase):
    return value.to(device)
  if isinstance(value, tuple):
    return tuple(_move(v, device) for v in value)
  if isinstance(value, list):
    return [_move(v, device) for v in value]
  if hasattr(value, "__dict__"):
    # Generic plain-object container (e.g. RolloutStorage.Batch / .Transition):
    # rebuild field-wise so every tensor leaf lands on `device`.
    out = object.__new__(type(value))
    for key, val in vars(value).items():
      setattr(out, key, _move(val, device))
    return out
  return value


def _apply_learner_device(alg, rollout_device: str, learner_device: str) -> None:
  debug = os.environ.get("MJLAB_LEARNER_DEBUG")
  storage = alg.storage
  if getattr(storage, "_learner_device_hooked", False):
    return

  orig_generator = storage.mini_batch_generator
  seen = [0]

  def mini_batch_generator(*args, **kwargs):
    for batch in orig_generator(*args, **kwargs):
      if debug and seen[0] == 0:
        print(f"[dbg] batch type={type(batch).__name__}")
        obs = batch.observations
        print(
          f"[dbg] batch.observations type={type(obs)} device={getattr(obs, 'device', None)}"
        )
        for k, v in obs.items() if hasattr(obs, "items") else []:
          print(f"[dbg]   obs[{k}] type={type(v)} device={getattr(v, 'device', None)}")
          break
        moved = _move(batch, learner_device)
        print(f"[dbg] moved obs device={getattr(moved.observations, 'device', None)}")
      seen[0] += 1
      yield _move(batch, learner_device)

  storage.mini_batch_generator = mini_batch_generator
  storage._learner_device_hooked = True

  # PPO.update() feeds update_normalization() straight from cpu-resident storage;
  # move that one input at the boundary instead of moving the storage.
  for module_name in ("actor", "critic"):
    module = getattr(alg, module_name)
    orig_update_normalization = module.update_normalization

    def update_normalization(obs, _orig=orig_update_normalization):
      return _orig(_move(obs, learner_device))

    module.update_normalization = update_normalization

  # Adaptive-KL controller numerics: the KL scalar is a pure function of the two
  # distribution parameterizations (Gaussian closed form, no module state), but
  # f32 log-prob reductions on MPS systematically overestimate it, which pins
  # the LR schedule at its 1e-5 floor (4000-iter evidence,
  # docs/results/worklog-mac-training-speed.md). Compute the control scalar in
  # f64 on CPU: the controller becomes numerically identical across devices, at
  # the cost of ~1MB of parameter transfers per minibatch.
  actor = alg.actor
  dist = actor.distribution

  def get_kl_divergence_f64(old_params, new_params):
    # NOTE: MPS tensors cannot cast straight to float64 ("MPS framework doesn't
    # support float64"); move to CPU in the source dtype first, then upcast.
    old64 = tuple(t.detach().to("cpu").to(torch.float64) for t in old_params)
    new64 = tuple(t.detach().to("cpu").to(torch.float64) for t in new_params)
    # The result must go back as f32: MPS has no float64 storage.
    return dist.kl_divergence(old64, new64).to("cpu", torch.float32).to(new_params[0].device)

  actor.get_kl_divergence = get_kl_divergence_f64

  orig_update = alg.update

  def update():
    # Optimizer state is created on `learner_device` at the first update and
    # stays there; parameters migrate per update, so the two always meet on
    # `learner_device` while the optimizer is touched.
    alg.actor.to(learner_device)
    alg.critic.to(learner_device)
    if debug:
      print(f"[dbg] actor param device={next(alg.actor.parameters()).device}")
      norm = getattr(alg.actor, "obs_normalizer", None)
      if norm is not None:
        print(f"[dbg] normalizer _mean device={norm._mean.device}")
    if getattr(alg, "symmetry", None) is not None:
      alg.symmetry.to(learner_device)
    try:
      return orig_update()
    finally:
      alg.actor.to(rollout_device)
      alg.critic.to(rollout_device)
      if getattr(alg, "symmetry", None) is not None:
        alg.symmetry.to(rollout_device)

  alg.update = update


def apply_learner_device(runner, learner_device: str) -> None:
  """Split the runner's PPO learner onto `learner_device` (experimental).

  No-op when `learner_device` matches the runner device or is unavailable.
  """
  rollout_device = runner.device
  if learner_device == rollout_device:
    return
  if learner_device == "mps" and not torch.backends.mps.is_available():
    print(
      f"[WARN] MJLAB_LEARNER_DEVICE={learner_device}: MPS unavailable, learner stays on {rollout_device}."
    )
    return
  _apply_learner_device(runner.alg, rollout_device, learner_device)
  print(
    f"[INFO] Learner device split: rollout={rollout_device}, learner={learner_device}"
  )
