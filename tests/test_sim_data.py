"""Tests for sim data bridge."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from conftest import get_test_device
from mujoco_qd._src.warp_util import from_data

from mjlab.sim.sim_data import QdBridge, TorchArray


@dataclass
class MockData:
  arr: object
  val: float = 1.0


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def mock_data():
  return MockData(arr=from_data(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)))


def test_torch_array_shares_memory():
  """TorchArray modifications affect underlying qd array."""
  qd_arr = from_data(np.array([1.0, 2.0], dtype=np.float32))
  torch_arr = TorchArray(qd_arr)
  torch_arr[0] = 99.0
  assert qd_arr.to_numpy()[0] == 99.0


def test_torch_array_ops_work():
  """TorchArray supports PyTorch operations."""
  qd_arr = from_data(np.array([1.0, 2.0], dtype=np.float32))
  torch_arr = TorchArray(qd_arr)
  assert torch.allclose(torch_arr * 2, torch.tensor([2.0, 4.0]))
  assert torch.sum(torch_arr).item() == 3.0  # type: ignore


def test_torch_array_expands_batched_singleton_world_dim():
  """Size-1 world dim expands to nworld when the engine marks the array batched."""
  qd_arr = from_data(np.array([[1.0, 2.0]], dtype=np.float32))
  setattr(qd_arr, "_is_batched", True)  # noqa: B010
  torch_arr = TorchArray(qd_arr, nworld=4)
  assert torch_arr.shape == (4, 2)


def test_torch_array_keeps_unbatched_singleton_dim():
  """A real size-1 leading dim without the batched marker is not expanded."""
  qd_arr = from_data(np.array([[1.0, 2.0]], dtype=np.float32))
  torch_arr = TorchArray(qd_arr, nworld=4)
  assert torch_arr.shape == (1, 2)


def test_bridge_wraps_arrays(mock_data):
  """QdBridge wraps qd arrays as TorchArray."""
  bridge = QdBridge(mock_data)
  assert isinstance(bridge.arr, TorchArray)
  assert bridge.val == 1.0  # Non-arrays pass through.


def test_bridge_preserves_memory_on_slice_assign(mock_data):
  """Slice assignment writes through to the underlying qd array."""
  bridge = QdBridge(mock_data)

  bridge.arr[:] = torch.zeros((2, 2))

  assert torch.all(bridge.arr._tensor == 0)
  assert np.all(mock_data.arr.to_numpy() == 0)


def test_bridge_raises_on_setattr(mock_data):
  """Direct assignment raises AttributeError with helpful message."""
  bridge = QdBridge(mock_data)

  with pytest.raises(AttributeError, match="Cannot set attribute 'arr' on QdBridge"):
    bridge.arr = torch.zeros((2, 2))

  with pytest.raises(AttributeError, match="Use in-place operations instead"):
    bridge.val = 42.0


def test_bridge_caches_wrappers(mock_data):
  """QdBridge caches wrapper objects."""
  bridge = QdBridge(mock_data)
  arr1 = bridge.arr
  arr2 = bridge.arr
  assert arr1 is arr2  # Same cached object.
