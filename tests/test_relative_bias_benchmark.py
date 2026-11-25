"""CPU control-flow fixtures only; no candidate GPU correctness claim."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.relative_bias import benchmark as b, adapter


class NativeRPCFixture:
    def __init__(self, *, compile_at=None, fail_at=None, mismatch_at=None, missing_rank=False):
        self.compile_at = compile_at; self.fail_at = fail_at; self.mismatch_at = mismatch_at
        self.missing_rank = missing_rank; self.capture = False; self.calls = []; self.invocations = 0
        self.phase = None; self.prepare_count = 0

    def __call__(self, fn, *args):
        self.calls.append((fn.__name__, args, self.capture))
        if fn is adapter.worker_prepare:
            self.prepare_count += 1; self.phase = args[0]
            return [{"rank": rank, "history_preparation_variant": "native"} for rank in (0, 1)]
        if fn is adapter.worker_helper_validation:
            return [{"rank": rank, "passed": True if self.phase == "prefill" else None,
                     "applicable": self.phase == "prefill"} for rank in (0, 1)]
        if fn is adapter.worker_helper_microbench:
            return [{"rank": rank, "reports": [], "applicable": self.phase == "prefill"} for rank in (0, 1)]
        if fn is b.base.worker_capture:
            self.capture = args[0]
            return [{"rank": rank, "capture_active": self.capture, "return_code": 0} for rank in (0, 1)]
        if fn is adapter.worker_evidence:
            return [{"rank": rank, "checkpoint": {key: self.invocations for key in
                      ("restore_count", "verified_restore_count", "verified_write_boundary_count")},
                      "compilation_monitor": {key: True for key in
                      ("native_jit_monitor_active", "native_jit_callback", "triton_autotuner_bench")}}
                    for rank in (0, 1)]
        if fn is not adapter.worker_iteration:
            raise AssertionError("Unexpected fixture callback")
        variant, stage, iteration, attempt, block, capture_inputs = args
        key = (block, variant, stage, iteration, attempt)
        if key == self.fail_at:
            raise RuntimeError("fixture native rank failed")
        self.invocations += 1
        return [{"rank": rank, "variant": variant, "stage": stage, "iteration": iteration,
                 "attempt": attempt, "block_index": block,
                 "wall_seconds": (0.01 if variant == "native" else 0.009)+rank*0.001,
                 "cuda_event_ms": 8+rank, "helper_calls": 8 if self.phase == "prefill" else 0,
                 "correctness": {"passed": key != self.mismatch_at, "logits_present": rank == 0,
                                 "reference_sha256": "a"*64, "reference_variant": "native"},
                 "compilation_events": [{"event": "fixture observed compilation"}] if key == self.compile_at else [],
                 "capture_active": self.capture}
                for rank in ((0,) if self.missing_rank else (0, 1))]


class RelativeBiasProtocolTests(unittest.TestCase):
    def run_fixture(self, rpc=None, phase="prefill", profile=False):
        rpc = rpc or NativeRPCFixture()
        saved = []
        case = b.run_case(rpc, phase, 256, list(range(4097)), Path("unused-cpu-fixture"), profile,
                          lambda c: saved.append(copy.deepcopy(c)))
        return case, rpc, saved

    def test_two_blocks_reverse_start_order_and_share_one_preparation(self):
        case, rpc, _ = self.run_fixture()
        self.assertEqual(rpc.prepare_count, 1)
        self.assertEqual(rpc.invocations, 62)  #2preparation +2*(5+10)*2variants.
        self.assertTrue(case["completed"])
        self.assertTrue(case["eligible_for_unprofiled_timing"])
        self.assertEqual([block["attempts"][0]["measurements"][0]["order"] for block in case["blocks"]],
                         [["native", "fused"], ["fused", "native"]])
        for block in case["blocks"]:
            attempt = block["attempts"][0]
            self.assertEqual(len(attempt["warmups"]), 5)
            self.assertEqual(len(attempt["measurements"]), 10)
            self.assertEqual(attempt["summary"]["variants"]["native"]["two_rank_completion_seconds"]["count"], 10)

    def test_native_reference_is_checked_across_variants(self):
        rpc = NativeRPCFixture(); rpc.phase = "prefill"
        observations = {name: b.base.combine_iteration(rpc(adapter.worker_iteration, name, "measured", 0, 0, 0, False), 1)
                        for name in ("native", "fused")}
        observations["fused"]["ranks"][0]["correctness"]["reference_sha256"] = "b"*64
        pair = b.combine_pair(observations, ("native", "fused"), False, require_helper=True)
        self.assertFalse(pair["correctness_passed"])
        self.assertFalse(pair["native_reference_matches"])

    def test_decode_zero_helper_calls_are_valid_unaffected_controls(self):
        case, rpc, _ = self.run_fixture(phase="decode")
        self.assertTrue(case["completed"])
        self.assertTrue(all(r["applicable"] is False for r in case["captured_input_correctness"]))
        pair = case["blocks"][0]["attempts"][0]["measurements"][0]
        self.assertTrue(pair["correctness_passed"])
        self.assertTrue(all(r["helper_calls"] == 0 for v in pair["observations"].values() for r in v["ranks"]))
        self.assertFalse(b.combine_pair(pair["observations"], pair["order"], False, require_helper=True)["correctness_passed"])

    def test_compilation_retains_ten_pairs_and_repeats_entire_block_set(self):
        case, rpc, _ = self.run_fixture(NativeRPCFixture(compile_at=(0, "fused", "measured", 3, 0)))
        self.assertTrue(case["completed"])
        self.assertEqual(case["blocks"][0]["accepted_attempt"], 1)
        self.assertEqual([len(a["measurements"]) for a in case["blocks"][0]["attempts"]], [10, 10])
        self.assertEqual(rpc.invocations, 92)

    def test_profile_capture_wraps_only_measured_pairs_and_never_headline(self):
        case, rpc, _ = self.run_fixture(profile=True)
        self.assertTrue(case["completed"])
        self.assertFalse(case["eligible_for_unprofiled_timing"])
        self.assertEqual([args[0] for name, args, _ in rpc.calls if name == "worker_capture"], [True, False, True, False])
        for name, args, active in rpc.calls:
            if name == "worker_iteration":
                self.assertEqual(active, args[1] == "measured")

    def test_partial_failure_retains_pairs_and_closes_capture(self):
        rpc = NativeRPCFixture(fail_at=(0, "fused", "measured", 2, 0)); saved = []
        with self.assertRaisesRegex(RuntimeError, "rank failed"):
            b.run_case(rpc, "prefill", 256, list(range(4097)), Path("unused"), True,
                       lambda c: saved.append(copy.deepcopy(c)))
        self.assertEqual(saved[-1]["status"], "failed")
        self.assertEqual(len(saved[-1]["blocks"][0]["attempts"][0]["measurements"]), 2)
        self.assertFalse(rpc.capture)

    def test_numerical_failure_is_preserved_and_cannot_be_eligible(self):
        case, _, _ = self.run_fixture(NativeRPCFixture(mismatch_at=(1, "fused", "measured", 4, 0)))
        self.assertFalse(case["completed"])
        self.assertFalse(case["eligible_for_unprofiled_timing"])
        self.assertEqual(len(case["blocks"][1]["attempts"][0]["measurements"]), 10)

    def test_missing_rank_rejected_before_measurements(self):
        with self.assertRaisesRegex(ValueError, "each of TP ranks"):
            self.run_fixture(NativeRPCFixture(missing_rank=True))

    def test_plan_predeclares_primary_and_two_blocks_without_gpu_imports(self):
        protocol = b.plan()
        self.assertEqual(protocol["confirmation_blocks"], 2)
        self.assertEqual(protocol["predeclared_adoption_gate"]["primary"], "prefill-L4096")
        self.assertEqual(len(protocol["cases"]), 6)
        self.assertEqual(protocol["original_helper_sha256"], adapter.HELPER_SHA256)


if __name__ == "__main__":
    unittest.main()
