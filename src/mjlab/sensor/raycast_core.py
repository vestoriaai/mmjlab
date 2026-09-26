"""General CPU ray-intersection core for the classic backend.

「几何场景 → 最近命中」的向量化求交层，供 Grid/Pinhole/Ring 三种射线模式统一
使用（Grid 高度扫描退化为特例）。替代 :mod:`mjlab.sensor.raycast_cpu` 的
grid 专用 hfield 扫描在三种模式下的角色：

- mesh：模型编译期一次性建 CPU BVH（每 mesh 一棵，中值分割，内容寻址缓存）；
  射线先做 geom 级世界 AABB 剪枝（逐环境），候选射线变换到 geom 局部系后走
  向量化 Möller–Trumbore（射线×三角批量，BVH 叶级收集，分块防内存爆炸）。
- plane / hfield：复用 :class:`~mjlab.sensor.raycast_cpu.CpuRaycastContext`
  的编译内核（组合而非替换；hfield 仍只支持局部系竖直射线）。
- 语义对齐 ``RayCastData``：无命中 distances=-1 / normals=0 / hit_pos=起点；
  命中距离为沿（单位）射线方向的参数 t，法线为绕序几何法线。

mesh 求交为单面（背面剔除，Möller–Trumbore det>0），与 mujoco_warp sensor
路径一致（``ray_mesh_with_bvh`` 的 ``cull_backfaces=True``）。注意 C 引擎
``mj_ray`` 是双面的：两者对「射线从封闭实体外部发射」的首命中等价，射线起点
在实体内部时 CPU 核心与 warp 一致地报告穿透（C 端会报背面出射距离）。

与 F-track v1 相同的范围限制：box/sphere 等解析图元不参与求交（保持移出射线
体积或用 backend='warp'）；hfield 斜射线 raise；静态几何位姿与 hfield 数据按
模板快照。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.sensor.raycast_cpu import CpuRaycastContext, _geom_invisible

_BIG = 1e30
_MT_EPS = 1e-12
_INV_TINY = 1e-12

_MESH_LEAF_SIZE = 8
_BVH_MAX_DEPTH = 14
# LIFO 栈槽数（2 的幂）：平衡中值分割深度 ≤ ceil(log2(T/leaf)) + 1，余量充足。
_STACK_SLOTS = 32
# 遍历迭代硬上限：覆盖 2k 三角网格全遍历（2K-1 ≈ 1020）；正常最近命中路径
# 几十次内经剪枝收敛，早退检查每轮触发，超限视为异常并 raise（不静默出错）。
_MAX_TRAVERSE_ITERS = 1024
_COMPACT_EVERY = 32  # 定期把已完成射线紧凑掉，防个别长尾射线拖住整块
_CHUNK = 65536  # 候选射线分块（防 [R, W] 中间张量内存爆炸）
_FULL_LEAF_MAX_RAYS = 8192  # 低于此批量叶处理走全宽（launch 受限区）
_LEAF_ARANGE = torch.arange(_MESH_LEAF_SIZE)


def _safe_inv(d: torch.Tensor) -> torch.Tensor:
  """倒数，零分量替换为带符号极小量（slab 测试无 NaN/Inf）。"""
  tiny = torch.where(d < 0, -_INV_TINY, _INV_TINY)
  d = torch.where(d.abs() < _INV_TINY, tiny, d)
  return d.reciprocal()


# ---------------------------------------------------------------------------
# BVH（numpy 构建，torch 张量托管）。
# ---------------------------------------------------------------------------


@dataclass
class MeshBVH:
  """线性化二叉 BVH：内部节点两个子节点，叶节点指向面连续区间。

  打包布局（减少遍历时逐节点 gather 次数）：node_bound = [bmin, bmax] 拼接，
  node_meta = [left, right, start, count] 拼接（叶的 left/right 为 -1，内部的
  start/count 为 -1）。
  """

  node_bound: torch.Tensor  # [K, 6] f32 = [bmin(3), bmax(3)]
  node_meta: torch.Tensor  # [K, 4] i64 = [left, right, start, count]
  face_order: torch.Tensor  # [T] i64（按叶区间重排后的面下标）


def _build_bvh_numpy(
  tri_bmin: np.ndarray, tri_bmax: np.ndarray
) -> tuple[np.ndarray, ...]:
  """中值分割 BVH（迭代式）。返回 (bmin, bmax, left, right, start, count, order)。"""
  T = len(tri_bmin)
  leaf = _MESH_LEAF_SIZE
  nodes: dict[int, tuple] = {}  # nid -> (bmin, bmax, left, right, start, count)
  order = np.empty(T, dtype=np.int64)
  fill = 0
  counter = 1
  work: list[tuple[np.ndarray, int, int]] = [(np.arange(T, dtype=np.int64), 0, 0)]
  while work:
    idx, depth, nid = work.pop()
    bmin = tri_bmin[idx].min(axis=0)
    bmax = tri_bmax[idx].max(axis=0)
    if len(idx) <= leaf or depth >= _BVH_MAX_DEPTH:
      nodes[nid] = (bmin, bmax, -1, -1, fill, len(idx))
      order[fill : fill + len(idx)] = idx
      fill += len(idx)
      continue
    cent = (tri_bmin[idx] + tri_bmax[idx]) * 0.5
    ext = cent.max(axis=0) - cent.min(axis=0)
    axis = int(np.argmax(ext))
    # 严格对半切（最宽质心轴排序取中位）：保证平衡 → 深度 ≤ log2(T)，栈容量
    # 有界。质心聚簇时 `<= median` 式分割可能极度偏斜，撑爆定容栈。
    perm = np.argsort(cent[:, axis], kind="stable")
    half = len(idx) // 2
    left_mask = np.zeros(len(idx), dtype=bool)
    left_mask[perm[:half]] = True
    li = idx[left_mask]
    ri = idx[~left_mask]
    lid = counter
    counter += 1
    rid = counter
    counter += 1
    nodes[nid] = (bmin, bmax, lid, rid, -1, -1)
    work.append((li, depth + 1, lid))
    work.append((ri, depth + 1, rid))
  K = counter
  rows = [nodes[i] for i in range(K)]
  bound = np.empty((K, 6), dtype=np.float32)
  meta = np.empty((K, 4), dtype=np.int64)
  for i, (bmin, bmax, left, right, start, count) in enumerate(rows):
    bound[i, :3] = bmin
    bound[i, 3:] = bmax
    meta[i, 0] = left
    meta[i, 1] = right
    meta[i, 2] = start
    meta[i, 3] = count
  return bound, meta, order


# ---------------------------------------------------------------------------
# MeshCollider：单 mesh 的求交器（BVH + 预展开三角形），内容寻址缓存。
# ---------------------------------------------------------------------------

_COLLIDER_CACHE: "OrderedDict[str, MeshCollider]" = OrderedDict()
_COLLIDER_CACHE_MAX = 64


def _mesh_cache_key(verts: np.ndarray, faces: np.ndarray) -> str:
  h = hashlib.sha1()
  h.update(f"v{verts.shape}f{faces.shape}".encode())
  h.update(np.ascontiguousarray(verts, dtype=np.float32).tobytes())
  h.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
  return h.hexdigest()


class MeshCollider:
  """单个 mesh 的 CPU 求交器：BVH + 叶级批量 Möller–Trumbore。

  三角形数据按叶区间重排（v0/e1/e2/n），使遍历的叶访问是连续 gather。
  """

  def __init__(
    self,
    verts: np.ndarray,
    faces: np.ndarray,
    bvh: MeshBVH,
    tri_v0: torch.Tensor,
    tri_e1: torch.Tensor,
    tri_e2: torch.Tensor,
    tri_n: torch.Tensor,
  ):
    self.verts = verts
    self.faces = faces
    self.bvh = bvh
    self.tri_v0 = tri_v0
    self.tri_e1 = tri_e1
    self.tri_e2 = tri_e2
    self.tri_n = tri_n

  @classmethod
  def from_mesh(
    cls, verts: torch.Tensor | np.ndarray, faces: torch.Tensor | np.ndarray
  ) -> "MeshCollider":
    """构建（或从内容寻址缓存取）一个 mesh 求交器。"""
    if isinstance(verts, torch.Tensor):
      verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
      faces = faces.detach().cpu().numpy()
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    key = _mesh_cache_key(verts, faces)
    cached = _COLLIDER_CACHE.get(key)
    if cached is not None:
      _COLLIDER_CACHE.move_to_end(key)
      return cached

    tri = verts[faces]  # [F, 3, 3]
    tri_bmin = tri.min(axis=1)
    tri_bmax = tri.max(axis=1)
    (bound, meta, order) = _build_bvh_numpy(tri_bmin, tri_bmax)
    faces_o = faces[order]
    tri_o = verts[faces_o]
    v0 = tri_o[:, 0]
    e1 = tri_o[:, 1] - v0
    e2 = tri_o[:, 2] - v0
    nrm = np.cross(e1, e2)
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(norm > 0.0, norm, 1.0)
    bvh = MeshBVH(
      node_bound=torch.from_numpy(np.ascontiguousarray(bound)),
      node_meta=torch.from_numpy(np.ascontiguousarray(meta)),
      face_order=torch.from_numpy(order),
    )
    collider = cls(
      verts=verts,
      faces=np.ascontiguousarray(faces_o, dtype=np.int64),
      bvh=bvh,
      tri_v0=torch.from_numpy(np.ascontiguousarray(v0, dtype=np.float32)),
      tri_e1=torch.from_numpy(np.ascontiguousarray(e1, dtype=np.float32)),
      tri_e2=torch.from_numpy(np.ascontiguousarray(e2, dtype=np.float32)),
      tri_n=torch.from_numpy(np.ascontiguousarray(nrm, dtype=np.float32)),
    )
    _COLLIDER_CACHE[key] = collider
    while len(_COLLIDER_CACHE) > _COLLIDER_CACHE_MAX:
      _COLLIDER_CACHE.popitem(last=False)
    return collider

  # 求交。

  def closest_hit(
    self, lo: torch.Tensor, ld: torch.Tensor, max_dist: float
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """geom 局部系射线 → 最近命中。

    Args:
      lo: [R, 3] 局部系射线起点。
      ld: [R, 3] 局部系射线方向。
      max_dist: 距离上限。

    Returns:
      (t [R]（无命中 +inf）, normal [R, 3]（单位、绕序法线；无命中 0））。
    """
    R = lo.shape[0]
    t_best = torch.full((R,), float("inf"), dtype=lo.dtype)
    n_best = torch.zeros(R, 3, dtype=lo.dtype)
    for r0 in range(0, R, _CHUNK):
      r1 = min(r0 + _CHUNK, R)
      self._traverse_chunk(lo[r0:r1], ld[r0:r1], max_dist, t_best, n_best, r0)
    return t_best, n_best

  def _traverse_chunk(
    self,
    lo: torch.Tensor,
    ld: torch.Tensor,
    max_dist: float,
    t_best: torch.Tensor,
    n_best: torch.Tensor,
    offset: int,
  ) -> None:
    """向量化的 LIFO 栈 BVH 遍历（最近命中，段剪枝 + 近子枝优先）。

    每轮：弹出栈顶节点 → 段-AABB 相交测试（以 best_t/max_dist 裁剪段）→
    叶则批量 Möller–Trumbore（单面），内部节点则按子节点入段距离排序压栈
    （远子枝先入栈，近子枝下一轮先弹）。全部射线完成后把命中合并回主缓冲。

    叶处理两种模式：小批量（launch 受限）直接全宽算；大批量（带宽受限）先
    nonzero 紧凑到命中叶的射线子集再算。周期性紧凑把已完成射线移出活跃集。
    """
    S = _STACK_SLOTS
    bvh = self.bvh
    num_faces = bvh.face_order.shape[0]
    node_bound = bvh.node_bound
    node_meta = bvh.node_meta
    R0 = lo.shape[0]
    stack = torch.full((R0, S), -1, dtype=torch.long)
    stack[:, 0] = 0
    sp = torch.zeros(R0, dtype=torch.long)
    best_t = torch.full((R0,), float("inf"), dtype=lo.dtype)
    best_n = torch.zeros(R0, 3, dtype=lo.dtype)
    # kept: 当前活跃张量到原始 chunk 下标的映射（紧凑时收缩；None = 全体）。
    kept: torch.Tensor | None = None
    inv = _safe_inv(ld)
    ld_active = ld
    lo_active = lo
    full_leaf = R0 <= _FULL_LEAF_MAX_RAYS

    for it in range(_MAX_TRAVERSE_ITERS):
      R = lo_active.shape[0]
      active = sp >= 0
      if not bool(active.any()):
        break
      s = sp & (S - 1)
      cur = stack.gather(1, s.unsqueeze(1)).squeeze(1)
      cur = torch.where(active, cur, torch.zeros_like(cur))

      bound = node_bound[cur]  # [R, 6]
      meta = node_meta[cur]  # [R, 4]
      nbmin = bound[:, :3]
      nbmax = bound[:, 3:]
      nstart = meta[:, 2]
      ncount = meta[:, 3]
      is_leaf = nstart >= 0

      # 段 vs 节点 AABB（slab），以 [0, min(best_t, max_dist)] 裁剪。
      t0 = (nbmin - lo_active) * inv
      t1 = (nbmax - lo_active) * inv
      tmin_n = torch.minimum(t0, t1).amax(-1)
      tmax_n = torch.maximum(t0, t1).amin(-1)
      enter = tmin_n.clamp(min=0.0)
      exit_t = torch.minimum(tmax_n, best_t.clamp(max=max_dist))
      node_hit = active & (enter <= exit_t)

      # ---- 叶：批量 Möller–Trumbore（单面）----
      leaf_hit = node_hit & is_leaf
      if bool(leaf_hit.any()):
        if full_leaf:
          sel_l = None
          start_l = nstart
          count_l = ncount.clamp(min=0)
          lo_l = lo_active
          ld_l = ld_active
          prev_t = best_t
        else:
          sel_l = leaf_hit.nonzero(as_tuple=True)[0]
          meta_l = meta.index_select(0, sel_l)
          start_l = meta_l[:, 2]
          count_l = meta_l[:, 3].clamp(min=0)
          lo_l = lo_active.index_select(0, sel_l)
          ld_l = ld_active.index_select(0, sel_l)
          prev_t = best_t.index_select(0, sel_l)
        # 三角形数组已按叶区间重排：叶 [start, start+count) 即连续下标。
        tidx = start_l.clamp(min=0).unsqueeze(1) + _LEAF_ARANGE.unsqueeze(0)
        tidx = tidx.clamp(max=num_faces - 1)
        tmask = _LEAF_ARANGE.unsqueeze(0) < count_l.unsqueeze(1)
        if sel_l is not None:
          tmask = tmask & leaf_hit[sel_l].unsqueeze(1)
        v0 = self.tri_v0[tidx]
        e1 = self.tri_e1[tidx]
        e2 = self.tri_e2[tidx]
        ld_u = ld_l.unsqueeze(1)
        lo_u = lo_l.unsqueeze(1)
        pvec = torch.cross(ld_u.expand_as(v0), e2, dim=-1)
        det = (e1 * pvec).sum(-1)
        ok = tmask & (det > _MT_EPS)
        inv_det = torch.where(ok, 1.0 / det.clamp(min=_MT_EPS), torch.zeros_like(det))
        tvec = lo_u.expand_as(v0) - v0
        u = (tvec * pvec).sum(-1) * inv_det
        qvec = torch.cross(tvec, e1, dim=-1)
        v = (ld_u.expand_as(v0) * qvec).sum(-1) * inv_det
        t = (e2 * qvec).sum(-1) * inv_det
        ok = (
          ok
          & (u >= 0.0)
          & (v >= 0.0)
          & (u + v <= 1.0)
          & (t >= 0.0)
          & (t < prev_t.unsqueeze(1))
          & (t <= max_dist)
        )
        t = torch.where(ok, t, torch.full_like(t, float("inf")))
        tmin_leaf, arg = t.min(-1)
        upd = tmin_leaf < prev_t
        n_hit = self.tri_n[tidx.gather(1, arg.unsqueeze(1)).squeeze(1)]
        if sel_l is None:
          best_t = torch.where(upd, tmin_leaf, best_t)
          best_n = torch.where(upd.unsqueeze(1), n_hit, best_n)
        else:
          best_t[sel_l] = torch.where(upd, tmin_leaf, prev_t)
          sel_u = sel_l[upd]
          best_n[sel_u] = n_hit[upd]

      # ---- 内部节点：子节点按入段距离排序压栈 ----
      push = node_hit & ~is_leaf
      if bool(push.any()):
        meta_c = node_meta[cur]
        left = meta_c[:, 0]
        right = meta_c[:, 1]
        bound_l = node_bound[left]
        bound_r = node_bound[right]
        enter_l = _child_enter(lo_active, inv, bound_l[:, :3], bound_l[:, 3:])
        enter_r = _child_enter(lo_active, inv, bound_r[:, :3], bound_r[:, 3:])
        near = torch.where(enter_l <= enter_r, left, right)
        far = torch.where(enter_l <= enter_r, right, left)
        neg = torch.full_like(near, -1)
        near = torch.where(push, near, neg)
        far = torch.where(push, far, neg)
        # LIFO：far 落在刚弹出的槽位，near 落在其上（下一轮先弹近子枝）。
        flat_base = torch.arange(R) * S
        stack.view(-1)[flat_base + s] = far
        stack.view(-1)[flat_base + ((s + 1) & (S - 1))] = near
        sp = torch.where(push, sp + 1, sp - 1)
      else:
        sp = sp - 1
      sp = sp.clamp(min=-1)

      # 长尾紧凑：把已完成射线的命中写回主缓冲后移出活跃集。
      if not full_leaf and (it + 1) % _COMPACT_EVERY == 0 and R > 1:
        keep = (sp >= 0).nonzero(as_tuple=True)[0]
        done = (sp < 0).nonzero(as_tuple=True)[0]
        if done.numel() > 0:
          sel_done = _select(kept, done, offset)
          _write_back(t_best, n_best, best_t, best_n, sel_done, done)
        if keep.numel() == 0:
          return  # 全部完成且已写回
        stack = stack.index_select(0, keep)
        sp = sp.index_select(0, keep)
        inv = inv.index_select(0, keep)
        lo_active = lo_active.index_select(0, keep)
        ld_active = ld_active.index_select(0, keep)
        best_t = best_t.index_select(0, keep)
        best_n = best_n.index_select(0, keep)
        kept = keep if kept is None else kept.index_select(0, keep)

    if bool((sp >= 0).any()):
      raise RuntimeError(
        "BVH traversal exceeded its iteration budget; this indicates a "
        "pathological mesh or a traversal bug."
      )
    remaining = torch.arange(sp.shape[0])
    sel = _select(kept, remaining, offset)
    _write_back(t_best, n_best, best_t, best_n, sel, remaining)


def _select(
  kept: torch.Tensor | None, local_idx: torch.Tensor, offset: int
) -> torch.Tensor:
  """局部活跃下标 → 主缓冲全局下标。"""
  sel = kept[local_idx] if kept is not None else local_idx
  return sel + offset


def _write_back(
  t_best: torch.Tensor,
  n_best: torch.Tensor,
  part_t: torch.Tensor,
  part_n: torch.Tensor,
  sel: torch.Tensor,
  local_idx: torch.Tensor,
) -> None:
  """把分块结果按 improved 掩码合并回主缓冲（sel 与 local_idx 等长）。"""
  prev_t = t_best[sel]
  upd = part_t[local_idx] < prev_t
  t_best[sel] = torch.where(upd, part_t[local_idx], prev_t)
  prev_n = n_best[sel]
  n_sel = part_n[local_idx]
  n_best[sel] = torch.where(upd.unsqueeze(1), n_sel, prev_n)


def _child_enter(
  lo: torch.Tensor, inv: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor
) -> torch.Tensor:
  """子节点 AABB 的入段距离（未命中 +inf）。"""
  t0 = (bmin - lo) * inv
  t1 = (bmax - lo) * inv
  tmin_n = torch.minimum(t0, t1).amax(-1)
  tmax_n = torch.maximum(t0, t1).amin(-1)
  return torch.where(
    tmax_n >= torch.clamp(tmin_n, min=0.0),
    tmin_n.clamp(min=0.0),
    torch.full_like(tmin_n, float("inf")),
  )


def intersect_trians_bruteforce(
  verts: torch.Tensor,
  faces: torch.Tensor,
  lo: torch.Tensor,
  ld: torch.Tensor,
  max_dist: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """参考实现：全三角形分块批量 Möller–Trumbore（单面），无 BVH。

  语义与 :meth:`MeshCollider.closest_hit` 一致（t 无命中 +inf，法线单位绕序），
  用于遍历正确性的内部对拍。
  """
  tri = verts[faces.long()]
  v0 = tri[:, 0]
  e1 = tri[:, 1] - v0
  e2 = tri[:, 2] - v0
  nrm = torch.cross(e1, e2, dim=-1)
  nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp(min=1e-12)
  R = lo.shape[0]
  F = v0.shape[0]
  t_best = torch.full((R,), float("inf"), dtype=lo.dtype)
  n_best = torch.zeros(R, 3, dtype=lo.dtype)
  lo_u = lo.unsqueeze(1)
  ld_u = ld.unsqueeze(1)
  for f0 in range(0, F, 256):
    v0c = v0[f0 : f0 + 256].unsqueeze(0)
    e1c = e1[f0 : f0 + 256].unsqueeze(0)
    e2c = e2[f0 : f0 + 256].unsqueeze(0)
    pvec = torch.cross(ld_u, e2c, dim=-1)
    det = (e1c * pvec).sum(-1)
    ok = det > _MT_EPS
    inv_det = torch.where(ok, 1.0 / det.clamp(min=_MT_EPS), torch.zeros_like(det))
    tvec = lo_u - v0c
    u = (tvec * pvec).sum(-1) * inv_det
    qvec = torch.cross(tvec, e1c, dim=-1)
    v = (ld_u * qvec).sum(-1) * inv_det
    t = (e2c * qvec).sum(-1) * inv_det
    ok = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= max_dist)
    t = torch.where(ok, t, torch.full_like(t, float("inf")))
    tmin_c, arg = t.min(-1)
    upd = tmin_c < t_best
    t_best = torch.where(upd, tmin_c, t_best)
    n_hit = nrm[f0 + arg]
    n_best = torch.where(upd.unsqueeze(1), n_hit, n_best)
  return t_best, n_best


# ---------------------------------------------------------------------------
# 场景级上下文：mesh 候选 + F-track hfield/plane 内核的组合。
# ---------------------------------------------------------------------------


@dataclass
class _MeshCandidate:
  geom_id: int
  body_id: int
  pose_mode: str  # "identity" | "constant" | "dynamic"（沿用 CpuRaycastContext）
  collider: MeshCollider
  pos0: torch.Tensor | None  # constant 模式的常量变换（模板烘焙）
  rot0: torch.Tensor | None
  aabb_center: torch.Tensor  # [3] geom 局部系 AABB
  aabb_half: torch.Tensor  # [3]
  # identity/constant 模式下预计算的世界 AABB（[1,3]，逐环境相同）。
  world_center_static: torch.Tensor | None
  world_half_static: torch.Tensor | None
  # 该 geom 是否落在某传感器帧的父 body 上（逐射线排除表 [N]）。
  ray_excluded: torch.Tensor | None


class RaycastCoreContext:
  """Per-sensor 通用射线求交上下文：mesh BVH 候选 + F-track hfield/plane 内核。

  候选选择与 :class:`CpuRaycastContext` 一致（geom group、透明剔除、
  ``flg_static=True``），并新增 mesh geom；逐射线 body 排除对 mesh 生效。
  """

  def __init__(self, mj_model: mujoco.MjModel, sensor, data) -> None:
    self._mj_model = mj_model
    self._data = data
    self._max_distance = float(sensor.cfg.max_distance)
    self._rays_per_frame = sensor.num_rays_per_frame

    self._inner = CpuRaycastContext(mj_model, sensor, data)
    # 既有 Grid 高度扫描入口的兼容委托（既有测试直接访问）。
    self.height_scan = self._inner.height_scan
    self._height_scan_fn = self._inner._height_scan_fn

    # 逐射线 body 排除表：frame -> body id。
    frame_body = torch.tensor(
      [bid for _, _, bid in sensor._frame_infos], dtype=torch.long
    )
    N = sensor.num_rays
    frame_of_ray = torch.arange(N).div(self._rays_per_frame, rounding_mode="floor")
    self._ray_frame_body = frame_body[frame_of_ray]  # [N]
    self._exclude_parent = bool(sensor.cfg.exclude_parent_body)

    # 模板位姿（复用 F-track 的分类器）。
    scratch = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, scratch)
    groups = sensor.cfg.include_geom_groups
    include_all = groups is None
    include = set(range(mujoco.mjNGROUP)) if include_all else set(groups)

    self.mesh_colliders: list[_MeshCandidate] = []
    for g in range(mj_model.ngeom):
      if int(mj_model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        continue
      if not include_all:
        group = int(np.clip(int(mj_model.geom_group[g]), 0, mujoco.mjNGROUP - 1))
        if group not in include:
          continue
      if _geom_invisible(mj_model, g):
        continue
      body_id = int(mj_model.geom_bodyid[g])
      pose_mode, pos0, rot0 = self._inner._classify_pose(mj_model, scratch, g, body_id)
      hid = int(mj_model.geom_dataid[g])
      vadr = int(mj_model.mesh_vertadr[hid])
      vnum = int(mj_model.mesh_vertnum[hid])
      fadr = int(mj_model.mesh_faceadr[hid])
      fnum = int(mj_model.mesh_facenum[hid])
      verts = np.array(mj_model.mesh_vert[vadr : vadr + vnum])
      faces = np.array(mj_model.mesh_face[fadr : fadr + fnum])
      collider = MeshCollider.from_mesh(verts, faces)
      tv = torch.from_numpy(np.ascontiguousarray(verts, dtype=np.float32))
      aabb_min = tv.amin(0)
      aabb_max = tv.amax(0)
      aabb_center = (aabb_min + aabb_max) * 0.5
      aabb_half = (aabb_max - aabb_min) * 0.5 + 1e-6
      wc = wh = None
      if pose_mode == "identity":
        wc = aabb_center.unsqueeze(0)
        wh = aabb_half.unsqueeze(0)
      elif pose_mode == "constant":
        assert pos0 is not None and rot0 is not None
        c_w = (aabb_center.unsqueeze(0) @ rot0.transpose(0, 1)) + pos0
        h_w = aabb_half.unsqueeze(0) @ rot0.abs().transpose(0, 1)
        wc = c_w
        wh = h_w
      ray_excluded = None
      if self._exclude_parent and bool((frame_body == body_id).any()):
        ray_excluded = self._ray_frame_body == body_id
      self.mesh_colliders.append(
        _MeshCandidate(
          geom_id=g,
          body_id=body_id,
          pose_mode=pose_mode,
          collider=collider,
          pos0=pos0,
          rot0=rot0,
          aabb_center=aabb_center,
          aabb_half=aabb_half,
          world_center_static=wc,
          world_half_static=wh,
          ray_excluded=ray_excluded,
        )
      )

  # ------------------------------------------------------------------
  # 统一入口。
  # ------------------------------------------------------------------

  def closest_hit(
    self,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """「几何场景 → 最近命中」。

    Args:
      rays_o: [B, N, 3] 世界系射线起点。
      rays_d: [B, N, 3] 世界系射线方向（单位）。

    Returns:
      (distances [B, N], normals_w [B, N, 3], hit_pos_w [B, N, 3])；
      无命中 distances=-1 / normals=0 / hit_pos=起点（语义对齐 RayCastData）。
    """
    B, N, _ = rays_o.shape
    has_mesh = bool(self.mesh_colliders)
    has_hp = bool(self._inner._hfields or self._inner._planes)

    if not has_mesh:
      if has_hp:
        distances, normals = self._inner.height_scan(rays_o, rays_d)
      else:
        distances = torch.full((B, N), -1.0, dtype=rays_o.dtype)
        normals = torch.zeros(B, N, 3, dtype=rays_o.dtype)
      hit_pos = rays_o + rays_d * distances.clamp(min=0.0).unsqueeze(-1)
      return distances, normals, hit_pos

    mesh_t = torch.full((B, N), _BIG, dtype=rays_o.dtype)
    mesh_n = torch.zeros(B, N, 3, dtype=rays_o.dtype)
    for cand in self.mesh_colliders:
      self._intersect_mesh(cand, rays_o, rays_d, mesh_t, mesh_n)

    mesh_valid = mesh_t < _BIG
    if has_hp:
      d_hp, n_hp = self._inner.height_scan(rays_o, rays_d)
      take = mesh_valid & ((d_hp < 0.0) | (mesh_t < d_hp))
      distances = torch.where(take, mesh_t, d_hp)
      normals = torch.where(take.unsqueeze(-1), mesh_n, n_hp)
    else:
      distances = torch.where(mesh_valid, mesh_t, torch.full_like(mesh_t, -1.0))
      normals = mesh_n
    hit_pos = rays_o + rays_d * distances.clamp(min=0.0).unsqueeze(-1)
    return distances, normals, hit_pos

  # ------------------------------------------------------------------
  # mesh 候选处理：geom AABB 剪枝 → 局部系 BVH 遍历 → 最近合并。
  # ------------------------------------------------------------------

  def _intersect_mesh(
    self,
    cand: _MeshCandidate,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    mesh_t: torch.Tensor,
    mesh_n: torch.Tensor,
  ) -> None:
    B, N, _ = rays_o.shape
    xmat: torch.Tensor | None = None
    if cand.pose_mode == "dynamic":
      xpos = self._data.geom_xpos[:, cand.geom_id]
      xmat = self._data.geom_xmat[:, cand.geom_id]
      if xmat.dim() == 2:
        xmat = xmat.reshape(B, 3, 3)
      # 世界 AABB：world = R @ local（行向量形式 local @ Rᵀ）；[B,3]。
      center = (cand.aabb_center @ xmat.transpose(1, 2)) + xpos
      half = cand.aabb_half @ xmat.abs().transpose(1, 2)
    else:
      center = cand.world_center_static
      half = cand.world_half_static
      assert center is not None and half is not None

    inv_d = _safe_inv(rays_d)
    t0 = ((center - half).unsqueeze(1) - rays_o) * inv_d
    t1 = ((center + half).unsqueeze(1) - rays_o) * inv_d
    tmin_n = torch.minimum(t0, t1).amax(-1)
    tmax_n = torch.maximum(t0, t1).amin(-1)
    enter = tmin_n.clamp(min=0.0)
    exit_t = tmax_n.clamp(max=self._max_distance)
    mask = (enter <= exit_t) & (tmax_n >= 0.0)
    if cand.ray_excluded is not None:
      mask = mask & ~cand.ray_excluded
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
      return

    o_flat = rays_o.reshape(-1, 3)
    d_flat = rays_d.reshape(-1, 3)
    env = torch.div(idx, N, rounding_mode="floor")
    if cand.pose_mode == "dynamic":
      assert xmat is not None
      xsel = xmat[env]
      lo = (o_flat[idx] - xpos[env]).unsqueeze(1) @ xsel
      ld = d_flat[idx].unsqueeze(1) @ xsel
    elif cand.pose_mode == "constant":
      assert cand.pos0 is not None and cand.rot0 is not None
      lo = (o_flat[idx] - cand.pos0).unsqueeze(1) @ cand.rot0
      ld = d_flat[idx].unsqueeze(1) @ cand.rot0
    else:
      lo = o_flat[idx]
      ld = d_flat[idx]
    lo = lo.reshape(-1, 3)
    ld = ld.reshape(-1, 3)

    # 跨环境精确去重：候选掩码逐环境一致（平移复制布局的典型情形）且各环境
    # 局部射线逐位相同时，只遍历 env0 的射线，结果平铺回全批量（精确，非近似）。
    tiled = False
    if B > 1:
      Nc = lo.shape[0] // B
      if Nc > 0 and Nc * B == lo.shape[0] and bool((mask == mask[0:1]).all()):
        lo_b = lo.view(B, Nc, 3)
        ld_b = ld.view(B, Nc, 3)
        if bool((lo_b == lo_b[0]).all()) and bool((ld_b == ld_b[0]).all()):
          lo = lo_b[0]
          ld = ld_b[0]
          tiled = True

    t, n_local = cand.collider.closest_hit(lo, ld, self._max_distance)
    if tiled:
      t = t.repeat(B)
      n_local = n_local.repeat(B, 1)
    prev = mesh_t.view(-1)[idx]
    improved = t < prev
    mesh_t.view(-1)[idx] = torch.where(improved, t, prev)
    if cand.pose_mode != "identity":
      # 法线变换与射线变换相反：world = R @ local（行向量形式 local @ Rᵀ）。
      if cand.pose_mode == "dynamic":
        assert xmat is not None
        rot_n = xmat[env]
      else:
        assert cand.rot0 is not None
        rot_n = cand.rot0
      n_local = (n_local.unsqueeze(1) @ rot_n.transpose(-1, -2)).reshape(-1, 3)
    n_world = n_local / n_local.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    mesh_n.view(-1, 3)[idx[improved]] = n_world[improved]
