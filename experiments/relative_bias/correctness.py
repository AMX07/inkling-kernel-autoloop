"""Untimed, byte-exact correctness gates for the relative-bias candidate.

Nothing runs on import. The full GPU suite must be run explicitly in the
authorized two-B300 experiment; a CPU oracle is not GPU validation.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .kernel import FLOAT_FORMATS, NATIVE_SOURCE_SHA256, unshear_rel_logits


def compile_offline():
    """Compile SM103 artifacts on CPU, explicitly without loading/launching CUDA.

    This catches frontend/PTX errors, not device correctness or performance.
    An explicit target avoids Triton's driver.active.get_current_target().
    API source: https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from .kernel import _load_kernel

    function = _load_kernel()
    results = []
    for dtype_name, (width, sign, magnitude, infinity, finite) in FLOAT_FORMATS.items():
        for requests, metadata_type in ((1, "i32"), (4, "i64")):
            constants = {"S0": 24 * 1280, "S1": 1280, "S2": 1,
                         "CACHE_STRIDE": 1 if requests == 1 else 2,
                         "CU_STRIDE": 1 if requests == 1 else 2,
                         "NUM_HEADS": 24, "NREQ": requests, "EXTENT": 1024, "SHEARED_WIDTH": 1280,
                         "SIGN_MASK": sign, "MAGNITUDE_MASK": magnitude,
                         "INF_BITS": infinity, "MAX_FINITE_BITS": finite, "BLOCK": 1024}
            signature = {name: "constexpr" for name in constants}
            signature.update(ShearedBits=f"*i{width}", CacheSeqLens="*" + metadata_type,
                             CuSeqLensQ="*" + metadata_type, OutputBits=f"*i{width}")
            compiled = triton.compile(ASTSource(fn=function, signature=signature, constexprs=constants),
                                      target=GPUTarget("cuda", 103, 32), options={"num_warps": 4})
            cubin = compiled.asm["cubin"]
            ptx = compiled.asm["ptx"].encode()
            results.append({"dtype": dtype_name, "requests": requests, "metadata_type": metadata_type,
                            "target": "cuda:103:32", "cubin_bytes": len(cubin),
                            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                            "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
                            "signature": signature, "constants": constants})
    return {"status": "compiled_only", "gpu_execution": False, "cuda_driver_context_requested": False,
            "correctness_claim": False, "artifacts": results}


def load_native_reference(source_path):
    """Load only the exact pinned helper AST, without importing native vLLM.

    Useful for CPU tests. Production correctness should pass the actual
    original callable captured by the model adapter instead.
    """
    import torch

    source = Path(source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != NATIVE_SOURCE_SHA256:
        raise ValueError("Native attention source does not match the frozen SHA256")
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_unshear_rel_logits"]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one native helper")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["_unshear_rel_logits"]


def sanitize_bits(bits, dtype_name):
    """Scalar CPU oracle for IEEE nan_to_num, with no float conversion."""
    width, sign, magnitude, infinity, finite = FLOAT_FORMATS[dtype_name]
    bits &= (1 << width) - 1
    value = bits & magnitude
    if value > infinity:
        return 0
    if value == infinity:
        return (bits & sign) | finite
    return bits


def scalar_column(token, distance, query_lengths, cache_lengths):
    """Independent request-by-request native indexing oracle for CPU tests."""
    start = 0
    for q_len, cache_len in zip(query_lengths, cache_lengths, strict=True):
        if start <= token < start + q_len:
            local = token - start
            past = cache_len - q_len
            tile_end = min((local // 128 + 1) * 128, q_len)
            n_max = (past + tile_end + 127) // 128
            return past + local - distance - (n_max - 10) * 128
        start += q_len
    raise ValueError("Token has no nonempty request")


def _bytes(tensor):
    """Return logical C-order bytes from an explicitly unit-stride CPU buffer.

    ``contiguous()`` may return a singleton or empty view with a non-unit
    stride: those dimensions do not affect its contiguity classification. A
    dtype-changing byte view still requires a unit last stride. Allocating the
    flattened destination also handles scalar inputs without changing dtype
    or performing floating arithmetic (NaN payloads and signed zero survive).
    This copy belongs only to untimed validation.
    """
    from experiments.context_sweep.state import logical_byte_view

    return logical_byte_view(tensor, device="cpu").numpy().tobytes()


def tensor_sha256(tensor):
    return hashlib.sha256(_bytes(tensor)).hexdigest()


def _metadata(sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent):
    """Explicitly synchronizes for validation; never used inside timing/capture."""
    starts = cu_seqlens_q.detach().cpu().tolist()
    cache = cache_seqlens.detach().cpu().tolist()
    if not starts or starts[0] != 0 or starts[-1] != num_tokens:
        raise AssertionError("Cumulative queries must start at0 and end at T")
    queries = [b - a for a, b in zip(starts, starts[1:])]
    if len(queries) != len(cache) or any(q < 0 or c < q for q, c in zip(queries, cache)):
        raise AssertionError("Invalid native sequence metadata")
    return {"num_tokens": num_tokens, "rel_extent": rel_extent,
            "input_shape": list(sheared.shape), "input_stride": list(sheared.stride()),
            "input_storage_offset": sheared.storage_offset(), "dtype": str(sheared.dtype),
            "metadata_dtype": [str(cache_seqlens.dtype), str(cu_seqlens_q.dtype)],
            "metadata_stride": [list(cache_seqlens.stride()), list(cu_seqlens_q.stride())],
            "query_lengths": queries, "cache_lengths": cache}


def _compare(expected, actual, label):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError(f"{label}: shape, dtype or device changed")
    if not actual.is_contiguous():
        raise AssertionError(f"{label}: candidate output is not contiguous")
    reference_bytes, candidate_bytes = _bytes(expected), _bytes(actual)
    if reference_bytes != candidate_bytes:
        mismatches = sum(a != b for a, b in zip(reference_bytes, candidate_bytes))
        raise AssertionError(f"{label}: byte-exact mismatch ({mismatches} differing bytes)")
    if not bool(torch.isfinite(actual).all().item()):
        raise AssertionError(f"{label}: nonfinite output survived nan_to_num")
    return {"bitwise_equal": True, "all_finite": True, "output_contiguous": True,
            "has_nonzero_finite_output": bool((actual != 0).any().item()),
            "output_shape": list(actual.shape), "output_stride": list(actual.stride()),
            "output_bytes": len(candidate_bytes),
            "output_sha256": hashlib.sha256(candidate_bytes).hexdigest()}


def assert_exact(native_helper, candidate=unshear_rel_logits, *, sheared, num_tokens,
                 cache_seqlens, cu_seqlens_q, rel_extent, label="captured"):
    """Compare actual native/candidate outputs and verify all inputs unchanged.

    Input/output copies and synchronization are intentional validation overhead,
    excluded from every benchmark timer. Fail closed on any mismatching byte.
    """
    info = _metadata(sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent)
    inputs = (sheared, cache_seqlens, cu_seqlens_q)
    before = [tensor_sha256(value) for value in inputs]
    args = (sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent)
    expected = native_helper(*args)
    if before != [tensor_sha256(value) for value in inputs]:
        raise AssertionError(f"{label}: native helper modified input buffers")
    actual = candidate(*args)
    if before != [tensor_sha256(value) for value in inputs]:
        raise AssertionError(f"{label}: candidate modified input buffers")
    comparison = _compare(expected, actual, label)
    return {"label": label, "status": "passed", "passed": True, **info, **comparison,
            "input_sha256": before, "inputs_unchanged": True}


def _special_bits(dtype_name):
    width, sign, _, infinity, finite = FLOAT_FORMATS[dtype_name]
    fraction_bits = {"float16": 10, "bfloat16": 7, "float32": 23}[dtype_name]
    smallest_normal = 1 << fraction_bits
    one = {"float16": 0x3C00, "bfloat16": 0x3F80, "float32": 0x3F800000}[dtype_name]
    values = [0, sign, 1, sign | 1, smallest_normal - 1, sign | (smallest_normal - 1),
              smallest_normal, sign | smallest_normal, one, sign | one, finite, sign | finite,
              infinity, sign | infinity, infinity | 1, sign | infinity | 1,
              infinity | (1 << (fraction_bits - 1)), sign | infinity | (1 << (fraction_bits - 1))]
    return [value if value < sign else value - (1 << width) for value in values]


def _make_case(query_lengths, past_lengths, extent, dtype_name, *, layout="contiguous",
               metadata_dtype="int32", heads=3, device="cuda"):
    import torch

    if len(query_lengths) != len(past_lengths):
        raise ValueError("Query/past lengths differ")
    tokens = sum(query_lengths)
    width = extent + 256
    extra_rows = 3 if layout in ("offset", "strided") else 0
    storage_heads = heads + 1 if layout == "strided" else heads
    storage_width = 2 * width + 3 if layout == "strided" else width
    shape = (tokens + 128 + extra_rows, storage_heads, storage_width)
    elements = shape[0] * shape[1] * shape[2]
    integer_dtype = torch.int32 if dtype_name == "float32" else torch.int16
    floating_dtype = getattr(torch, dtype_name)
    # All IEEE encodings are possible, including signaling NaNs. No floating
    # arithmetic ever constructs the fixtures or accidentally quiets a NaN.
    raw = (torch.arange(elements, dtype=torch.int64) * 2654435761 + 17).to(integer_dtype)
    specials = torch.tensor(_special_bits(dtype_name), dtype=integer_dtype)
    raw[::17] = specials[torch.arange(raw[::17].numel()) % specials.numel()]
    storage = raw.reshape(shape).view(floating_dtype).to(device)
    if layout == "strided":
        sheared = storage[3:, :heads, 1:1 + width * 2:2]
    elif layout == "offset":
        sheared = storage[3:]
    else:
        sheared = storage
    if layout == "broadcast":
        sheared = storage[:, :1, :].expand(-1, heads, -1)
    cu = [0]
    for value in query_lengths:
        cu.append(cu[-1] + value)
    cache = [q + p for q, p in zip(query_lengths, past_lengths, strict=True)]
    md_dtype = getattr(torch, metadata_dtype)

    def metadata(values):
        if layout == "strided":
            backing = torch.full((2 * len(values) + 1,), -999, dtype=md_dtype, device=device)
            view = backing[1::2]
            view.copy_(torch.tensor(values, dtype=md_dtype, device=device))
            return view
        return torch.tensor(values, dtype=md_dtype, device=device)

    return {"sheared": sheared, "num_tokens": tokens, "cache_seqlens": metadata(cache),
            "cu_seqlens_q": metadata(cu), "rel_extent": extent}


def adversarial_cases(device="cuda"):
    """Yield bounded fixtures; exact source-sized captures are supplied separately."""
    configurations = [
        ([], [], 1024, "contiguous", "int32"),
        ([0, 0, 0], [0, 128, 65536], 1024, "strided", "int64"),
        ([1], [0], 0, "contiguous", "int32"),
        ([1], [0], 1, "contiguous", "int32"),
        ([127], [1], 127, "offset", "int32"),
        ([128], [0], 1024, "contiguous", "int32"),
        ([129], [127], 1024, "strided", "int32"),
        ([255], [128], 1025, "offset", "int64"),
        ([256], [1023], 1031, "contiguous", "int32"),
        ([257], [65536], 2049, "strided", "int32"),
        ([0, 1, 0, 127, 128, 129, 0], [0, 1, 5, 127, 128, 1024, 65536], 1024, "strided", "int32"),
        ([128, 0, 0, 128], [65536, 0, 128, 2**31 + 513], 1024, "contiguous", "int64"),
        ([3, 1, 257, 2], [3, 5, 128, 65536], 1031, "broadcast", "int64"),
    ]
    for dtype_name in FLOAT_FORMATS:
        for index, (queries, past, extent, layout, md_dtype) in enumerate(configurations):
            label = f"adversarial/{dtype_name}/{index}/{layout}/{md_dtype}"
            yield label, _make_case(queries, past, extent, dtype_name, layout=layout,
                                   metadata_dtype=md_dtype, device=device)


def _graph_checks(native_helper, candidate, device):
    import torch

    results = []
    for dtype_name in FLOAT_FORMATS:
        for md_dtype in ("int32", "int64"):
            case = _make_case([0, 127, 0, 130], [0, 1, 0, 127], 1024, dtype_name,
                              metadata_dtype=md_dtype, layout="strided", device=device)
            args = tuple(case[key] for key in ("sheared", "num_tokens", "cache_seqlens", "cu_seqlens_q", "rel_extent"))
            input_hash = tensor_sha256(case["sheared"])
            pointers = [case[key].data_ptr() for key in ("sheared", "cache_seqlens", "cu_seqlens_q")]

            def capture(function):
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        function(*args)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = function(*args)
                return graph, output

            native_graph, native_output = capture(native_helper)
            candidate_graph, candidate_output = capture(candidate)
            versions = [([0, 127, 0, 130], [0, 1, 0, 127]),
                        ([128, 0, 129, 0], [65536, 128, 7, 0]),
                        ([1, 1, 0, 255], [129, 65536, 0, 513])]
            output_hashes = []
            for version, (queries, past) in enumerate(versions):
                cu = [0]
                for q in queries:
                    cu.append(cu[-1] + q)
                cache = [q + p for q, p in zip(queries, past, strict=True)]
                case["cache_seqlens"].copy_(torch.tensor(cache, dtype=case["cache_seqlens"].dtype, device=device))
                case["cu_seqlens_q"].copy_(torch.tensor(cu, dtype=case["cu_seqlens_q"].dtype, device=device))
                native_graph.replay()
                candidate_graph.replay()
                expected = native_helper(*args)
                label = f"graph/{dtype_name}/{md_dtype}/metadata-{version}"
                _compare(expected, native_output, label + "/native-capture")
                comparison = _compare(expected, candidate_output, label + "/candidate-capture")
                output_hashes.append(comparison["output_sha256"])
                if pointers != [case[key].data_ptr() for key in ("sheared", "cache_seqlens", "cu_seqlens_q")]:
                    raise AssertionError("Graph input buffer addresses changed")
                if tensor_sha256(case["sheared"]) != input_hash:
                    raise AssertionError("Graph execution modified the sheared input")
                if case["cache_seqlens"].cpu().tolist() != cache or case["cu_seqlens_q"].cpu().tolist() != cu:
                    raise AssertionError("Graph execution modified metadata")
                results.append({"label": label, "status": "passed", "passed": True, **comparison,
                                "metadata_updated_in_place": True, "native_graph_also_verified": True,
                                "input_addresses_unchanged": True, "inputs_unchanged": True,
                                "query_lengths": queries, "cache_lengths": cache})
            if len(set(output_hashes)) != len(versions):
                raise AssertionError("Graph fixtures did not expose every metadata update")
    return results


def run_correctness(native_helper, candidate=unshear_rel_logits, *, device="cuda", graph=True):
    """Run all adversarial eager cases and optional persistent-metadata graphs.

    A successful return is required before any performance claim. Exceptions
    must be preserved by the calling experiment rather than converted to skips.
    """
    import torch

    if torch.device(device).type != "cuda":
        raise ValueError("GPU correctness requires an explicit CUDA device")
    with torch.cuda.device(device):
        eager = []
        for label, case in adversarial_cases(device):
            try:
                result = assert_exact(native_helper, candidate, label=label, **case)
            except Exception as error:
                raise RuntimeError(f"{label}: correctness gate failed after {len(eager)} completed eager cases") from error
            if "/broadcast/" in label:
                result["nonzero_broadcast_values_required"] = True
                if result["has_nonzero_finite_output"] is not True:
                    raise AssertionError(label + ": broadcast fixture exercised only zero masking")
            eager.append(result)
        graphs = _graph_checks(native_helper, candidate, device) if graph else []
    return {"kind": "relative-bias-correctness-v1", "status": "passed", "passed": True,
            "native_source_sha256": NATIVE_SOURCE_SHA256,
            "candidate_source_sha256": hashlib.sha256(Path(__file__).with_name("kernel.py").read_bytes()).hexdigest(),
            "comparison": "exact output bytes, dtype/device/shape, contiguous layout, and unchanged inputs",
            "device": str(device), "eager_case_count": len(eager), "eager_cases": eager,
            "graph_checks_requested": graph, "graph_case_count": len(graphs), "graph_cases": graphs,
            "timing_scope": "Correctness/preparation only; no timing measurements."}
