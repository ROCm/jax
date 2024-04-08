# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PagedAttention TPU kernel."""

import functools
from typing import Optional

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,
      vmem_buffer,
      sem,
      page_indices,
      page_indices_start_offset,
      num_pages_to_load,
      head_index,
  ):
    self._vmem_buffer = vmem_buffer
    self._num_pages_to_load = num_pages_to_load
    if head_index is not None:
      self._pages_hbm_ref = pages_hbm_ref.at[head_index]
    else:
      self._pages_hbm_ref = pages_hbm_ref
    self._sem = sem
    self._page_indices = page_indices
    self._page_indices_start_offset = page_indices_start_offset
    self._async_copies = [
        self._make_async_copy(i) for i in range(self._num_pages_to_load)
    ]

  def _make_async_copy(self, i):
    page_index = self._page_indices[self._page_indices_start_offset + i]
    return pltpu.make_async_copy(
        self._pages_hbm_ref.at[page_index], self._vmem_buffer.at[i], self._sem
    )

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()

  def wait_and_get_loaded(self) -> jax.Array:
    """Wait async copies and gets the loaded buffer as a jax.Array."""
    for async_copy in self._async_copies:
      async_copy.wait()
    head_dim = self._vmem_buffer.shape[-1]
    return self._vmem_buffer[...].astype(jnp.float32).reshape(-1, head_dim)


def paged_flash_attention_kernel(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    sem,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    megacore_mode: str,
    program_ids=(),
):
  """Pallas kernel for paged attention."""
  if program_ids:
    core_index, b, h, i = program_ids
  else:
    core_index, b, h, i = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
        pl.program_id(3),
    )
  num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
  bk = page_size * pages_per_compute_block
  num_cores = pl.num_programs(0)

  b_step = num_cores if megacore_mode == "batch" else 1
  b_start = core_index if megacore_mode == "batch" else 0
  h_step = num_cores if megacore_mode == "kv_head" else 1
  h_start = core_index if megacore_mode == "kv_head" else 0

  h = h * h_step + h_start
  b = b * b_step + b_start
  length = lengths_ref[b]

  def compute_block_indices(b, h, i):

    def advance_b():
      next_b = b + b_step

      def advance_to_next_non_zero_length():
        next_next_b = next_b + b_step
        return lax.fori_loop(
            lax.div(next_next_b, b_step),
            lax.div(batch_size, b_step),
            lambda _, b: jnp.where(lengths_ref[b] == 0, b + b_step, b),
            next_next_b,
        )

      return (
          lax.cond(
              jnp.logical_and(next_b < batch_size, lengths_ref[next_b] == 0),
              advance_to_next_non_zero_length,
              lambda: next_b,
          ),
          h_start,
          0,
      )

    def advance_h():
      next_h = h + h_step
      return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

    return lax.cond(i * bk < lengths_ref[b], lambda: (b, h, i), advance_h)

  def create_kv_async_copy_descriptors(b, h, i, buffer_index):
    page_offset = b * pages_per_sequence + i * pages_per_compute_block
    pages_to_load = pages_per_compute_block
    async_copy_k = MultiPageAsyncCopyDescriptor(
        k_pages_hbm_ref,
        k_vmem_buffer.at[buffer_index],
        sem,
        page_indices_ref,
        page_offset,
        pages_to_load,
        h,
    )
    async_copy_v = MultiPageAsyncCopyDescriptor(
        v_pages_hbm_ref,
        v_vmem_buffer.at[buffer_index],
        sem,
        page_indices_ref,
        page_offset,
        pages_to_load,
        h,
    )
    return async_copy_k, async_copy_v

  @pl.when(i * bk < length)
  def flash_attention():
    step = step_ref[0]
    buffer_index = buffer_index_ref[0]

    @pl.when(i == 0)
    def init():
      m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
      l_ref[...] = jnp.zeros_like(l_ref)
      o_ref[...] = jnp.zeros_like(o_ref)

    @pl.when(step == 0)
    def prefetch_first_block():
      async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
          b, h, i, buffer_index
      )
      async_copy_k.start()
      async_copy_v.start()

    next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

    @pl.when(next_b < batch_size)
    def prefetch_next_block():
      next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
      async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
          next_b, next_h, next_i, next_buffer_index
      )
      async_copy_next_k.start()
      async_copy_next_v.start()
      buffer_index_ref[0] = next_buffer_index

    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(
        b, h, i, buffer_index
    )
    q = q_ref[...].astype(jnp.float32)
    k = async_copy_k.wait_and_get_loaded()
    qk = jnp.einsum('hd,td->ht', q, k, preferred_element_type=jnp.float32)
    mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])
    m_prev, l_prev = m_ref[...], l_ref[...]
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

    v = async_copy_v.wait_and_get_loaded()
    o_curr_times_l_curr = jnp.dot(s_curr, v)

    m_ref[...], l_ref[...] = m_next, l_next_safe
    o_ref[...] = (
        (l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe
    ).astype(o_ref.dtype)

    step_ref[0] = step + 1


def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    sem,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    megacore_mode: str,
):
  core_index, b, h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  def body(i, _):
    paged_flash_attention_kernel(
        lengths_ref,
        page_indices_ref,
        buffer_index_ref,
        step_ref,
        q_ref,
        k_pages_hbm_ref,
        v_pages_hbm_ref,
        o_ref,
        m_ref,
        l_ref,
        k_vmem_buffer,
        v_vmem_buffer,
        sem,
        batch_size=batch_size,
        pages_per_compute_block=pages_per_compute_block,
        pages_per_sequence=pages_per_sequence,
        mask_value=mask_value,
        megacore_mode=megacore_mode,
        program_ids=(core_index, b, h, i),
    )
    return ()

  bk = pages_per_compute_block * k_pages_hbm_ref.shape[-2]
  lax.fori_loop(0, lax.div(lengths_ref[b] + bk - 1, bk), body, ())


@functools.partial(
    jax.jit,
    static_argnames=[
        "pages_per_compute_block",
        "mask_value",
        "megacore_mode",
        "inline_seq_dim",
    ],
)
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    pages_per_compute_block: int,
    megacore_mode: Optional[str] = None,
    inline_seq_dim: bool = True,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Paged grouped query attention.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array the length of each example.
    page_indices: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
      should be in the range of [0, total_num_pages), indicating where to locate
      the page in `k_pages` or `v_pages`.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    pages_per_compute_block: how many pages to be processed in one flash
      attention block in the pallas kernel.
    megacore_mode: if set, enable megacore to parallelize the computation. Must
      be one of ['kv_head', 'batch', None]. Ceveat: set this only if megacore is
      enabled, otherwise the kernel may hang. If you are not sure, leave it to
      None.
      * None: disable megacore parallelism.
      * kv_head: megacore parallelism on KV heads; requires number of KV heads
        divisible by 2.
      * batch: megacore parallelism on batch dimension; requires batch divisible
        by 2.
    inline_seq_dim: whether to fuse kernel instances along the sequence dim into
      one kernel.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]).
  """
  batch_size, num_heads, head_dim = q.shape
  num_kv_heads, _, page_size, head_dim_k = k_pages.shape
  batch_size_paged_indices, pages_per_sequence = page_indices.shape

  if k_pages.shape != v_pages.shape:
    raise ValueError(
        f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
        f" {v_pages.shape}"
    )
  if num_heads % num_kv_heads != 0:
    raise ValueError(
        "Number of Q heads must be divisible by number of KV heads. Got"
        f" {num_heads} and {num_kv_heads}."
    )
  if head_dim_k != head_dim:
    raise ValueError(
        "head_dim of Q must be the same as that of K/V. Got"
        f" {head_dim} and {head_dim_k}."
    )
  if pages_per_sequence % pages_per_compute_block != 0:
    raise ValueError(
        "pages_per_compute_block must be divisible by pages per sequence. Got"
        f" {pages_per_compute_block} and {pages_per_sequence}."
    )
  if lengths.shape != (batch_size,):
    raise ValueError("`lengths` and `q` must have the same batch size")
  if batch_size_paged_indices != batch_size:
    raise ValueError("`page_indices` and `q` must have the same batch size")
  if lengths.dtype != jnp.int32:
    raise ValueError(
        "The dtype of `lengths` must be int32. Got {lengths.dtype}"
    )

  # TODO(dinghua): get the actual cores per chip once there's an official API.
  if megacore_mode == "kv_head":
    if num_kv_heads % 2 != 0:
      raise ValueError(
          "number of KV heads must be even when megacore_mode is 'kv_head'"
      )
    num_cores = 2
  elif megacore_mode == "batch":
    if batch_size % 2 != 0:
      raise ValueError("batch size must be even when megacore_mode is 'batch'")
    num_cores = 2
  elif megacore_mode is None:
    num_cores = 1
  else:
    raise ValueError("megacore_mode must be one of ['kv_head', 'batch', None]")

  if (num_heads // num_kv_heads) % 8 != 0:
    # Reshape q to hint XLA to pick a <1x128> layout otherwise it will pick a
    # <8x128> layout for a <1x128> memref inside the kernel and error out.
    q = q.reshape(batch_size, num_heads, 1, head_dim)
    if megacore_mode == "kv_head":
      q_block_spec = pl.BlockSpec(
          lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0, 0),
          (None, num_heads // num_kv_heads, None, head_dim),
      )
    elif megacore_mode == "batch":
      q_block_spec = pl.BlockSpec(
          lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0, 0),
          (None, num_heads // num_kv_heads, None, head_dim),
      )
    else:
      q_block_spec = pl.BlockSpec(
          lambda core_index, b, h, *_: (b, h, 0, 0),
          (None, num_heads // num_kv_heads, None, head_dim),
      )
    q_dtype_for_kernel_launch = jnp.float32
  else:
    if megacore_mode == "kv_head":
      q_block_spec = pl.BlockSpec(
          lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0),
          (None, num_heads // num_kv_heads, head_dim),
      )
    elif megacore_mode == "batch":
      q_block_spec = pl.BlockSpec(
          lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0),
          (None, num_heads // num_kv_heads, head_dim),
      )
    else:
      q_block_spec = pl.BlockSpec(
          lambda core_index, b, h, *_: (b, h, 0),
          (None, num_heads // num_kv_heads, head_dim),
      )
    q_dtype_for_kernel_launch = q.dtype

  if inline_seq_dim:
    kernel = paged_flash_attention_kernel_inline_seq_dim
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_kv_heads // num_cores
        if megacore_mode == "kv_head"
        else num_kv_heads,
    )  # type: ignore
    dimension_sematics = ("parallel", "arbitrary", "arbitrary")  # type: ignore
  else:
    kernel = paged_flash_attention_kernel
    grid = (
        num_cores,
        batch_size // num_cores if megacore_mode == "batch" else batch_size,
        num_kv_heads // num_cores
        if megacore_mode == "kv_head"
        else num_kv_heads,
        pages_per_sequence // pages_per_compute_block,
    )  # type: ignore
    dimension_sematics = ("parallel", "arbitrary", "arbitrary", "arbitrary")  # type: ignore

  out, _, _ = pl.pallas_call(
      functools.partial(
          kernel,
          pages_per_sequence=pages_per_sequence,
          batch_size=batch_size,
          pages_per_compute_block=pages_per_compute_block,
          mask_value=mask_value,
          megacore_mode=megacore_mode,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=[
              q_block_spec,
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
              pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
          ],
          out_specs=[
              q_block_spec,
              q_block_spec,
              q_block_spec,
          ],
          grid=grid,
          scratch_shapes=(
              pltpu.VMEM(
                  (
                      2,
                      pages_per_compute_block,
                      page_size,
                      head_dim,
                  ),
                  k_pages.dtype,
              ),  # k buffer
              pltpu.VMEM(
                  (
                      2,
                      pages_per_compute_block,
                      page_size,
                      head_dim,
                  ),
                  v_pages.dtype,
              ),  # v buffer
              pltpu.SemaphoreType.DMA,
          ),
      ),
      mosaic_params=dict(
          dimension_semantics=dimension_sematics,
      ),
      out_shape=[
          jax.ShapeDtypeStruct(q.shape, q_dtype_for_kernel_launch),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
          jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
      ],
  )(
      lengths,
      page_indices.reshape(-1),
      jnp.zeros((1,), jnp.int32),  # buffer index
      jnp.zeros((1,), jnp.int32),  # step
      q.astype(q_dtype_for_kernel_launch),
      k_pages,
      v_pages,
  )
  return out.reshape(batch_size, num_heads, head_dim).astype(q.dtype)