"""Exact one-kernel candidate for the pinned native ``_unshear_rel_logits``.

Native source: vllm/models/inkling2/ops/attention.py, SHA256 below, lines88–125.
Native production allocation is ``(T + 128, H, rel_extent + 256)`` contiguous;
the prefill call can pass a nonzero-storage-offset leading-row slice. This
candidate also supports arbitrary nonnegative element strides and int32/int64
metadata. All batch-dependent values are loaded from the supplied device
buffers on each invocation/replay. No values are read by the host wrapper.

Only index arithmetic and integer bit operations are used. In particular,
finite values (including signed zero and subnormals) do not round-trip through
another floating type. NaN becomes positive zero; infinities become signed
largest finite values of the original dtype, matching torch.nan_to_num.

The helper is experimental until the GPU correctness gates pass. It does not
change the SM103 attention fallback, quantization, or the attention algorithm.
Triton APIs: https://triton-lang.org/main/python-api/generated/triton.language.load.html
and https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
"""

from __future__ import annotations

NATIVE_SOURCE_SHA256 = "ff3164d9d02e1e317698e239d114796a091424f8df7bb1c310381127c7d4b1b1"
NATIVE_MODULE = "vllm.models.inkling2.ops.attention"
NATIVE_HELPER = "_unshear_rel_logits"
_TRITON_KERNEL = None

# IEEE encodings: width, sign mask, magnitude mask, infinity, largest finite.
FLOAT_FORMATS = {
    "float16": (16, 0x8000, 0x7FFF, 0x7C00, 0x7BFF),
    "bfloat16": (16, 0x8000, 0x7FFF, 0x7F80, 0x7F7F),
    "float32": (32, 0x80000000, 0x7FFFFFFF, 0x7F800000, 0x7F7FFFFF),
}


def _unshear_kernel(
    ShearedBits, CacheSeqLens, CuSeqLensQ, OutputBits,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    CACHE_STRIDE: tl.constexpr, CU_STRIDE: tl.constexpr,
    NUM_HEADS: tl.constexpr, NREQ: tl.constexpr,
    EXTENT: tl.constexpr, SHEARED_WIDTH: tl.constexpr,
    SIGN_MASK: tl.constexpr, MAGNITUDE_MASK: tl.constexpr,
    INF_BITS: tl.constexpr, MAX_FINITE_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per token/head/column tile. Native extent1024 uses one tile.
    row = tl.program_id(0).to(tl.int64)
    token = row // NUM_HEADS
    head = row % NUM_HEADS
    if NREQ == 1:
        req = tl.full((), 0, tl.int32)
    else:
        # Find first request whose END exceeds this token. Repeated cumulative
        # lengths represent zero-query requests and must be skipped.
        low = tl.full((), 0, tl.int32)
        high = tl.full((), NREQ, tl.int32)
        while low < high:
            mid = (low + high) // 2
            end = tl.load(CuSeqLensQ + (mid + 1) * CU_STRIDE).to(tl.int64)
            advance = end <= token
            low = tl.where(advance, mid + 1, low)
            high = tl.where(advance, high, mid)
        req = low
    start = tl.load(CuSeqLensQ + req * CU_STRIDE).to(tl.int64)
    end = tl.load(CuSeqLensQ + (req + 1) * CU_STRIDE).to(tl.int64)
    q_len = end - start
    past = tl.load(CacheSeqLens + req * CACHE_STRIDE).to(tl.int64) - q_len
    i_loc = token - start
    pos = past + i_loc
    tile_end = tl.minimum(((i_loc >> 7) + 1) * 128, q_len)
    n_max = (past + tile_end + 127) >> 7
    offset = (n_max - 10) * 128
    distance = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    column = pos - distance - offset
    valid = (distance < EXTENT) & (column >= 0) & (column < SHEARED_WIDTH)
    clamped = tl.minimum(tl.maximum(column, 0), SHEARED_WIDTH - 1)
    # The native helper clamps then overwrites invalid positions with +0.
    # Masked loads avoid reading those clamped values at all, with equal bits.
    bits = tl.load(ShearedBits + token * S0 + head * S1 + clamped * S2,
                   mask=valid, other=0).to(tl.uint32)
    magnitude = bits & MAGNITUDE_MASK
    replacement_inf = (bits & SIGN_MASK) | MAX_FINITE_BITS
    bits = tl.where(magnitude == INF_BITS, replacement_inf, bits)
    bits = tl.where(magnitude > INF_BITS, 0, bits)
    tl.store(OutputBits + row * EXTENT + distance, bits, mask=distance < EXTENT)


def _load_kernel():
    """Initialize Triton only on an actual candidate call, without launching."""
    global _TRITON_KERNEL, tl
    if _TRITON_KERNEL is None:
        import triton
        import triton.language as tl

        _TRITON_KERNEL = triton.jit(_unshear_kernel)
    return _TRITON_KERNEL


def unshear_rel_logits(sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent):
    """Drop-in candidate with exactly the native positional argument contract.

    Valid metadata has cu[0]=0, nondecreasing cu, cu[-1]=num_tokens, and
    cache_seqlens>=diff(cu); the caller owns these preconditions as in native
    vLLM. Shape/device/dtype checks use host descriptors only and do not sync.
    CUDA graph replay may update metadata contents in-place at fixed shapes.
    Returns a fresh, contiguous tensor with the input floating dtype/device.
    """
    import torch

    if type(num_tokens) is not int or type(rel_extent) is not int:
        raise TypeError("num_tokens and rel_extent must be Python integers")
    if num_tokens < 0 or rel_extent < 0:
        raise ValueError("num_tokens and rel_extent must be nonnegative")
    if sheared.ndim != 3 or sheared.shape[0] < num_tokens or sheared.shape[2] < 1:
        raise ValueError("sheared must have shape (at least T, heads, positive width)")
    dtype_name = str(sheared.dtype).removeprefix("torch.")
    if dtype_name not in FLOAT_FORMATS:
        raise TypeError("Only float16, bfloat16 and float32 are supported")
    if cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
        raise ValueError("sequence metadata must be one-dimensional")
    if cu_seqlens_q.numel() != cache_seqlens.numel() + 1:
        raise ValueError("cu_seqlens_q needs one more element than cache_seqlens")
    if num_tokens and cache_seqlens.numel() == 0:
        raise ValueError("nonempty queries require at least one request")
    if cache_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens_q.dtype not in (torch.int32, torch.int64):
        raise TypeError("metadata must use int32 or int64")
    if not sheared.is_cuda or cache_seqlens.device != sheared.device or cu_seqlens_q.device != sheared.device:
        raise ValueError("sheared and metadata must share one CUDA device")
    if any(stride < 0 for value in (sheared, cache_seqlens, cu_seqlens_q) for stride in value.stride()):
        raise ValueError("negative strides are unsupported")
    num_heads = sheared.shape[1]
    output = torch.empty((num_tokens, num_heads, rel_extent), dtype=sheared.dtype, device=sheared.device)
    if output.numel() == 0:
        return output
    width, sign, magnitude, infinity, finite = FLOAT_FORMATS[dtype_name]
    integer_dtype = torch.int16 if width == 16 else torch.int32
    kernel = _load_kernel()
    block = 1024
    kernel[(num_tokens * num_heads, (rel_extent + block - 1) // block)](
        sheared.view(integer_dtype), cache_seqlens, cu_seqlens_q, output.view(integer_dtype),
        *sheared.stride(), cache_seqlens.stride(0), cu_seqlens_q.stride(0),
        num_heads, cache_seqlens.numel(), rel_extent, sheared.shape[2],
        sign, magnitude, infinity, finite, block, num_warps=4,
    )
    return output


# Convenient same-name export for adapters, without altering the implementation.
_unshear_rel_logits = unshear_rel_logits
