"""Bridge for zero-copy PyTorch views over quadrants (qd) arrays.

The qd backend's counterpart of the former Warp bridge: simulation arrays are
exposed as torch tensors sharing memory through DLPack (write-through verified
on qd.cpu and qd.metal; quadrants 1.3.4 R4), so the entity/manager/sensor
stack reads and writes state without copies.

Note on device ordering: on CPU devices the qd kernels and torch consumers
see the same memory synchronously. On GPU devices (qd.metal) writes from one
framework become visible to the other only after a device sync
(``torch.mps.synchronize()``); callers stepping the sim already synchronize
at the framework boundary.
"""

from typing import Any, Generic, Tuple, TypeVar

import torch

T = TypeVar("T")

_QD_TO_TORCH = {
  "f16": torch.float16,
  "f32": torch.float32,
  "f64": torch.float64,
  "i8": torch.int8,
  "i16": torch.int16,
  "i32": torch.int32,
  "i64": torch.int64,
  "u8": torch.uint8,
  "u32": torch.uint32,
  "u64": torch.uint64,
}


def _qd_dtype_to_torch(dtype: Any) -> torch.dtype:
  name = str(dtype)
  try:
    return _QD_TO_TORCH[name]
  except KeyError:
    raise TypeError(f"Unsupported qd dtype for the torch bridge: {dtype}") from None


def _is_qd_array(val: Any) -> bool:
  """Duck-type check: qd ndarrays expose shape/dtype/to_dlpack."""
  return (
    not isinstance(val, torch.Tensor)
    and hasattr(val, "to_dlpack")
    and hasattr(val, "shape")
    and hasattr(val, "dtype")
  )


class TorchArray:
  """qd array that behaves like a torch.Tensor with shared memory."""

  def __init__(self, qd_array: Any, nworld: int | None = None) -> None:
    """Initialize the tensor proxy with a qd ndarray."""
    self._qd_array = qd_array
    shape = tuple(int(s) for s in qd_array.shape)
    if 0 in shape:
      # DLPack is undefined for zero-length extents; nothing to share anyway.
      self._tensor = torch.zeros(shape, dtype=_qd_dtype_to_torch(qd_array.dtype))
    else:
      self._tensor = torch.from_dlpack(qd_array.to_dlpack())

    if (
      nworld is not None
      and nworld > 1
      and len(self._tensor.shape) > 0
      and self._tensor.shape[0] == 1
      and (self._tensor.stride(0) == 0 or getattr(qd_array, "_is_batched", False))
    ):
      new_shape = (nworld,) + tuple(self._tensor.shape[1:])
      self._tensor = self._tensor.expand(new_shape)

  @property
  def qd_array(self) -> Any:
    return self._qd_array

  def __repr__(self) -> str:
    """Return string representation of the underlying tensor."""
    return repr(self._tensor)

  def __getitem__(self, idx: Any) -> Any:
    """Get item(s) from the tensor using standard indexing."""
    return self._tensor[idx]

  def __setitem__(self, idx: Any, value: Any) -> None:
    """Set item(s) in the tensor using standard indexing."""
    self._tensor[idx] = value

  def __getattr__(self, name: str) -> Any:
    """Delegate attribute access to the underlying tensor."""
    return getattr(self._tensor, name)

  @property
  def device(self) -> torch.device:
    return self._tensor.device

  @classmethod
  def __torch_function__(
    cls,
    func: Any,
    types: Tuple[type, ...],
    args: Tuple[Any, ...] = (),
    kwargs: Any = None,
  ) -> Any:
    """Intercept torch.* function calls to unwrap TorchArray objects."""
    if kwargs is None:
      kwargs = {}

    # Only intercept when at least one argument is our proxy.
    if not any(issubclass(t, cls) for t in types):
      return NotImplemented

    def _unwrap(x: Any) -> Any:
      """Unwrap TorchArray objects to their underlying tensors."""
      return x._tensor if isinstance(x, cls) else x

    # Unwrap all TorchArray objects in args and kwargs.
    unwrapped_args = tuple(_unwrap(arg) for arg in args)
    unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

    return func(*unwrapped_args, **unwrapped_kwargs)

  # Arithmetic operators.

  def __add__(self, other: Any) -> Any:
    return self._tensor + other

  def __radd__(self, other: Any) -> Any:
    return other + self._tensor

  def __sub__(self, other: Any) -> Any:
    return self._tensor - other

  def __rsub__(self, other: Any) -> Any:
    return other - self._tensor

  def __mul__(self, other: Any) -> Any:
    return self._tensor * other

  def __rmul__(self, other: Any) -> Any:
    return other * self._tensor

  def __truediv__(self, other: Any) -> Any:
    return self._tensor / other

  def __rtruediv__(self, other: Any) -> Any:
    return other / self._tensor

  def __pow__(self, other: Any) -> Any:
    return self._tensor**other

  def __rpow__(self, other: Any) -> Any:
    return other**self._tensor

  def __neg__(self) -> Any:
    return -self._tensor

  def __pos__(self) -> Any:
    return +self._tensor

  def __abs__(self) -> Any:
    return abs(self._tensor)

  # Comparison operators.

  def __eq__(self, other: Any) -> Any:
    return self._tensor == other

  def __ne__(self, other: Any) -> Any:
    return self._tensor != other

  def __lt__(self, other: Any) -> Any:
    return self._tensor < other

  def __le__(self, other: Any) -> Any:
    return self._tensor <= other

  def __gt__(self, other: Any) -> Any:
    return self._tensor > other

  def __ge__(self, other: Any) -> Any:
    return self._tensor >= other


class QdBridge(Generic[T]):
  """Wraps mujoco_qd objects to expose qd arrays as PyTorch tensors.

  Automatically converts qd array attributes to TorchArray objects
  on access, enabling direct PyTorch operations on simulation data.
  Recursively wraps nested structures that contain qd arrays.

  IMPORTANT: This wrapper is read-only. To modify array data, use
  in-place operations like `obj.field[:] = value`. Direct assignment
  like `obj.field = new_array` will raise an AttributeError to prevent
  accidental aliasing breakage (e.g. cached default views).
  """

  def __init__(self, struct: T, nworld: int | None = None) -> None:
    object.__setattr__(self, "_struct", struct)
    object.__setattr__(self, "_wrapped_cache", {})
    object.__setattr__(self, "_nworld", nworld)

  def __getattr__(self, name: str) -> Any:
    """Get attribute from the wrapped data, wrapping qd arrays as TorchArray."""
    if name == "device":
      # 消费方(如 raycast 常数张量的设备选择)需要与底层张量一致的设备;
      # 从任一 qd 数组的 DLPack 张量取(空数组也在 _wrapped_cache 之前处理)。
      cached = next(iter(self._wrapped_cache.values()), None)
      if cached is not None:
        return cached.device
      for f in vars(self._struct).values() if hasattr(self._struct, "__dict__") else ():
        if _is_qd_array(f) and f.shape and 0 not in tuple(f.shape):
          return TorchArray(f).device
      return torch.device("cpu")
    # Check cache first to avoid recreating wrappers.
    if name in self._wrapped_cache:
      return self._wrapped_cache[name]

    val = getattr(self._struct, name)

    # Wrap qd arrays.
    if _is_qd_array(val):
      wrapped = TorchArray(val, nworld=self._nworld)
      self._wrapped_cache[name] = wrapped
      return wrapped

    # Recursively wrap nested structures that contain qd arrays.
    if _contains_qd_arrays(val):
      wrapped = QdBridge(val, nworld=self._nworld)
      self._wrapped_cache[name] = wrapped
      return wrapped

    return val

  def __setattr__(self, name: str, value: Any) -> None:
    """Prevent attribute setting to preserve aliasing."""
    raise AttributeError(
      f"Cannot set attribute '{name}' on QdBridge. "
      f"This wrapper is read-only to preserve memory sharing. "
      f"Use in-place operations instead: obj.{name}[:] = value"
    )

  def __repr__(self) -> str:
    """Return string representation of the wrapped struct."""
    return f"QdBridge({repr(self._struct)})"

  @property
  def struct(self) -> T:
    """Access the underlying wrapped struct."""
    return self._struct

  def clear_cache(self) -> None:
    """Clear the wrapped cache to force re-wrapping of arrays.

    This should be called after operations that modify the underlying qd arrays,
    such as expand_model_fields(), to ensure the cache reflects the updated arrays.
    """
    object.__setattr__(self, "_wrapped_cache", {})


def _contains_qd_arrays(obj: Any) -> bool:
  """Check if an object or its attributes contain any qd arrays."""
  if _is_qd_array(obj):
    return True

  # Check if it's a struct-like object with attributes
  if hasattr(obj, "__dict__"):
    return any(
      _is_qd_array(getattr(obj, attr))
      for attr in dir(obj)
      if not attr.startswith("_")
    )

  return False


# Backwards-compatible alias for the former Warp bridge name.
WarpBridge = QdBridge
