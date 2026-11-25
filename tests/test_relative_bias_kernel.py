"""CPU checks; CUDA correctness is a separate mandatory experiment gate.

Run with uv: ``uv run --no-project python -m unittest tests.test_relative_bias_kernel``.
Torch-dependent tests run in the pinned runtime CPU preflight when unavailable
locally. Pure-Python tests still verify frozen source, IEEE semantics and indices.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

from experiments.relative_bias import correctness
from experiments.relative_bias import kernel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "planning/tokenspeed-baseline-source/vllm/models/inkling2/ops/attention.py"


class StructuralTests(unittest.TestCase):
    def test_native_source_hash_and_exact_callable_signature(self):
        source = SOURCE.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), kernel.NATIVE_SOURCE_SHA256)
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_unshear_rel_logits")
        self.assertEqual([arg.arg for arg in node.args.args], list(inspect.signature(kernel.unshear_rel_logits).parameters))

    def test_wrapper_never_reads_device_values_on_host(self):
        tree = ast.parse(inspect.getsource(kernel.unshear_rel_logits))
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"item", "tolist", "cpu", "numpy", "synchronize"})
        module = ast.parse(Path(kernel.__file__).read_text())
        top_level_imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all(not isinstance(node, ast.Import) for node in top_level_imports))

    def test_zero_query_requests_and_tile_boundaries(self):
        queries = [0, 127, 0, 128, 129, 0]
        caches = [0, 128, 0, 256, 1153, 0]
        self.assertEqual(correctness.scalar_column(126, 0, queries, caches), 1279)
        self.assertEqual(correctness.scalar_column(127, 0, queries, caches), 1152)
        self.assertEqual(correctness.scalar_column(255, 0, queries, caches), 1152)
        self.assertEqual(correctness.scalar_column(383, 0, queries, caches), 1152)
        cu = [0]
        for value in queries:
            cu.append(cu[-1] + value)
        # Compare device-style endpoint binary search against independent linear
        # request traversal across every token, including zero-length requests.
        for token in range(sum(queries)):
            low, high = 0, len(queries)
            while low < high:
                mid = (low + high) // 2
                if cu[mid + 1] <= token:
                    low = mid + 1
                else:
                    high = mid
            local = token - cu[low]
            past = caches[low] - queries[low]
            end = min(((local >> 7) + 1) * 128, queries[low])
            offset = (((past + end + 127) >> 7) - 10) * 128
            for distance in (0, 1, 127, 128, 1023, 2048):
                self.assertEqual(past + local - distance - offset,
                                 correctness.scalar_column(token, distance, queries, caches))

    def test_integer64_sequence_positions_do_not_overflow(self):
        queries = [0, 1, 129, 0]
        caches = [0, 2**31 + 514, 2**40 + 129, 0]
        for token in range(130):
            column = correctness.scalar_column(token, 0, queries, caches)
            self.assertGreaterEqual(column, 0)
            self.assertLess(column, 1280)

    def test_every_fp16_and_bf16_encoding(self):
        # struct performs independent IEEE decoding; comparison never uses the
        # candidate's magnitude classification for the expected classification.
        for dtype in ("float16", "bfloat16"):
            _, sign, _, infinity, finite = kernel.FLOAT_FORMATS[dtype]
            for bits in range(1 << 16):
                if dtype == "float16":
                    decoded = struct.unpack("<e", struct.pack("<H", bits))[0]
                else:
                    decoded = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                if math.isnan(decoded):
                    expected = 0
                elif math.isinf(decoded):
                    expected = finite if decoded > 0 else sign | finite
                else:
                    expected = bits
                self.assertEqual(correctness.sanitize_bits(bits, dtype), expected)

    def test_fp32_extremes_signed_zero_and_subnormals(self):
        values = correctness._special_bits("float32") + [0x7F800001, 0xFFC00001, 0x007FFFFF, 0x807FFFFF]
        values += [((i * 2654435761) + 17) & 0xFFFFFFFF for i in range(4096)]
        for bits in values:
            bits &= 0xFFFFFFFF
            decoded = struct.unpack("<f", struct.pack("<I", bits))[0]
            expected = 0 if math.isnan(decoded) else (0x7F7FFFFF if decoded > 0 else 0xFF7FFFFF) if math.isinf(decoded) else bits
            self.assertEqual(correctness.sanitize_bits(bits, "float32"), expected)

    @unittest.skipUnless(importlib.util.find_spec("triton"), "Triton checked in pinned CPU preflight")
    def test_lazy_jit_construction_needs_no_cuda_launch(self):
        jit = kernel._load_kernel()
        self.assertTrue(callable(jit.run))
        self.assertEqual(len(jit.arg_names), len(inspect.signature(kernel._unshear_kernel).parameters))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch checked in pinned CPU preflight")
class TorchReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.native = staticmethod(correctness.load_native_reference(SOURCE))

    def test_bytes_singleton_strided_metadata_regression(self):
        torch = self.torch
        for dtype_name, format_char in (("int32", "i"), ("int64", "q")):
            # This is the actual one-request strided fixture that failed the
            # first GPU correctness run, before either helper was compared.
            case = correctness._make_case([129], [127], 1024, "float16",
                                          layout="strided", metadata_dtype=dtype_name, device="cpu")
            metadata = case["cache_seqlens"]
            self.assertEqual(metadata.shape, (1,))
            self.assertEqual(metadata.stride(), (2,))
            self.assertGreater(metadata.storage_offset(), 0)
            self.assertTrue(metadata.is_contiguous())
            self.assertEqual(metadata.contiguous().stride(), (2,))
            with self.assertRaisesRegex(RuntimeError, "stride.*must be 1"):
                metadata.detach().contiguous().cpu().view(torch.uint8)
            expected = struct.pack("=" + format_char, 256)
            self.assertEqual(correctness._bytes(metadata), expected)
            self.assertEqual(correctness.tensor_sha256(metadata), hashlib.sha256(expected).hexdigest())
            result = correctness.assert_exact(self.native, self.native, label="singleton-strided-regression", **case)
            self.assertTrue(result["passed"])

    def test_bytes_preserves_ieee_encodings_in_strided_scalar_and_empty_inputs(self):
        torch = self.torch
        for dtype_name, (width, *_) in kernel.FLOAT_FORMATS.items():
            integer_dtype = torch.int32 if width == 32 else torch.int16
            floating_dtype = getattr(torch, dtype_name)
            # Exhaust every FP16/BF16 encoding, including signaling NaNs.
            # FP32 covers special encodings, signed zeros and varied payloads.
            values = (list(range(-(1 << 15), 1 << 15)) if width == 16
                      else correctness._special_bits(dtype_name) + [0x12345678, -123456789])
            raw = torch.tensor(values, dtype=integer_dtype)
            backing = torch.full((2 * len(values) + 1,), -17, dtype=integer_dtype)
            backing[1::2].copy_(raw)
            source = backing.view(floating_dtype)[1::2]
            expected = b"".join(value.to_bytes(width // 8, sys.byteorder, signed=True) for value in values)
            self.assertEqual(correctness._bytes(source), expected)
            self.assertEqual(correctness._bytes(source[:0]), b"")
            for index in (0, len(values) // 2, len(values) - 1):
                element_bytes = values[index].to_bytes(width // 8, sys.byteorder, signed=True)
                self.assertEqual(correctness._bytes(source[index]), element_bytes)
                self.assertEqual(correctness._bytes(source[index:index + 1]), element_bytes)
            # Logical order excludes storage holes and follows a transposed,
            # broadcast view rather than copying the underlying allocation.
            view = source[:6].reshape(2, 3).t().unsqueeze(-1).expand(3, 2, 2)
            expected_view = b"".join(values[i * 3 + j].to_bytes(width // 8, sys.byteorder, signed=True)
                                     for j in range(3) for i in range(2) for _ in range(2))
            self.assertEqual(correctness._bytes(view), expected_view)

    def test_every_adversarial_fixture_can_be_hashed_before_helper_comparison(self):
        count = 0
        for label, case in correctness.adversarial_cases(device="cpu"):
            for name in ("sheared", "cache_seqlens", "cu_seqlens_q"):
                tensor = case[name]
                with self.subTest(label=label, input=name):
                    self.assertEqual(len(correctness._bytes(tensor)), tensor.numel() * tensor.element_size())
            count += 1
        self.assertEqual(count, 39)

    def test_correctness_failure_retains_fixture_label_and_original_cause(self):
        failure = RuntimeError("byte hash failed")
        with mock.patch.object(self.torch.cuda, "device", return_value=nullcontext()), \
             mock.patch.object(correctness, "adversarial_cases", return_value=[("fixture-ok", {}), ("singleton-fixture", {})]), \
             mock.patch.object(correctness, "assert_exact", side_effect=[{"passed": True}, failure]):
            with self.assertRaisesRegex(RuntimeError, "singleton-fixture: correctness gate failed after 1 completed") as raised:
                correctness.run_correctness(self.native, self.native, device="cuda", graph=False)
        self.assertIs(raised.exception.__cause__, failure)

    def test_native_cpu_against_scalar_oracle_all_dtypes_and_strides(self):
        torch = self.torch
        for dtype in kernel.FLOAT_FORMATS:
            case = correctness._make_case([0, 1, 0, 129], [0, 127, 0, 2**31 + 513], 1031,
                                          dtype, layout="strided", metadata_dtype="int64", heads=1, device="cpu")
            got = self.native(**case)
            integer_dtype = torch.int32 if dtype == "float32" else torch.int16
            width = 32 if dtype == "float32" else 16
            raw = case["sheared"].view(integer_dtype).tolist()
            queries, caches = [0, 1, 0, 129], [0, 128, 0, 2**31 + 642]
            expected = []
            for token in range(case["num_tokens"]):
                head_values = []
                for distance in range(case["rel_extent"]):
                    column = correctness.scalar_column(token, distance, queries, caches)
                    bits = correctness.sanitize_bits(raw[token][0][column], dtype) if 0 <= column < case["sheared"].shape[-1] else 0
                    head_values.append(bits if bits < 1 << (width - 1) else bits - (1 << width))
                expected.append([head_values])
            expected = torch.tensor(expected, dtype=integer_dtype).view(getattr(torch, dtype))
            self.assertEqual(correctness._bytes(got), correctness._bytes(expected))
            self.assertTrue(got.is_contiguous())

    def test_native_empty_queries_and_zero_extent(self):
        for queries, past, extent in (([], [], 1024), ([0, 0], [0, 128], 1024), ([1], [0], 0)):
            case = correctness._make_case(queries, past, extent, "bfloat16", device="cpu")
            result = correctness.assert_exact(self.native, self.native, label="cpu-empty", **case)
            self.assertTrue(result["passed"])
            self.assertEqual(result["output_bytes"], 0)

    def test_broadcast_fixture_exercises_nonzero_values(self):
        # Extent257 would make these offsets entirely invalid; testing only
        # zero masking would miss a broken zero-head-stride implementation.
        for dtype in kernel.FLOAT_FORMATS:
            case = correctness._make_case([3, 1, 257, 2], [3, 5, 128, 65536], 1031,
                                          dtype, layout="broadcast", metadata_dtype="int64", device="cpu")
            self.assertEqual(case["sheared"].stride(1), 0)
            result = correctness.assert_exact(self.native, self.native, label="cpu-broadcast", **case)
            self.assertTrue(result["has_nonzero_finite_output"])

    def test_comparison_rejects_negative_zero_change(self):
        torch = self.torch
        with self.assertRaisesRegex(AssertionError, "byte-exact mismatch"):
            correctness._compare(torch.tensor([[[-0.0]]]), torch.tensor([[[0.0]]]), "signed-zero")

    def test_input_mutation_is_detected(self):
        case = correctness._make_case([1], [0], 1024, "float16", device="cpu")

        def mutating(*args):
            args[0].zero_()
            return self.native(*args)

        with self.assertRaisesRegex(AssertionError, "modified input buffers"):
            correctness.assert_exact(self.native, mutating, **case)

    def test_non_cuda_candidate_fails_without_loading_triton(self):
        case = correctness._make_case([1], [0], 1024, "float16", device="cpu")
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            kernel.unshear_rel_logits(**case)


if __name__ == "__main__":
    unittest.main()
