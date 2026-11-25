"""CPU orchestration tests. These do not claim native GPU correctness."""
import copy
import unittest

from experiments.context_sweep import benchmark as b


class FakeNativeRPC:
    def __init__(self, *, compile_at=None, fail_at=None, mismatch_at=None):
        self.compile_at = compile_at
        self.fail_at = fail_at
        self.mismatch_at = mismatch_at
        self.calls = []
        self.capture = False
        self.invocations = 0

    def __call__(self, fn, *args):
        self.calls.append((fn.__name__, args, self.capture))
        if fn is b.worker_prepare:
            return [{"rank": rank} for rank in (0, 1)]
        if fn is b.worker_capture:
            self.capture = args[0]
            return [{"rank": rank, "capture_active": self.capture, "return_code": 0} for rank in (0, 1)]
        if fn is b.worker_evidence:
            return [{"rank": rank, "checkpoint": {
                "restore_count": self.invocations, "verified_restore_count": self.invocations,
                "verified_write_boundary_count": self.invocations}, "compilation_monitor": {
                    "native_jit_monitor_active": True, "native_jit_callback": True,
                    "triton_autotuner_bench": True}} for rank in (0, 1)]
        self.assert_iteration(fn)
        stage, iteration, attempt = args
        key = (stage, iteration, attempt)
        if key == self.fail_at:
            raise RuntimeError("native rank failed")
        self.invocations += 1
        return [{"rank": rank, "stage": stage, "iteration": iteration, "attempt": attempt,
                 "wall_seconds": 0.01 + rank*0.002, "cuda_event_ms": 9+rank,
                 "memory": {"peak_allocated_bytes": 100+rank, "peak_reserved_bytes": 200+rank,
                            "incremental_peak_allocated_bytes": 10, "incremental_peak_reserved_bytes": 20},
                 "correctness": {"passed": key != self.mismatch_at, "logits_present": rank == 0},
                 "compilation_events": [{"event": "actual mocked JIT callback"}] if key == self.compile_at else [],
                 "capture_active": self.capture} for rank in (0, 1)]

    @staticmethod
    def assert_iteration(fn):
        if fn is not b.worker_iteration:
            raise AssertionError(f"Unexpected callback: {fn}")


class OrchestrationTests(unittest.TestCase):
    def run_fixture(self, rpc=None, profile=False):
        rpc = rpc or FakeNativeRPC()
        persisted = []
        case = b.run_case(rpc, "decode", 256, list(range(4097)), profile,
                          lambda c: persisted.append(copy.deepcopy(c)))
        return case, rpc, persisted

    def test_completion_metric_is_maximum_not_sum_and_requires_both_ranks(self):
        rpc = FakeNativeRPC()
        rows = rpc(b.worker_iteration, "measured", 0, 0)
        merged = b.combine_iteration(rows, 0.1)
        self.assertEqual(merged["two_rank_completion_seconds"], 0.012)
        self.assertNotEqual(merged["two_rank_completion_seconds"], sum(r["wall_seconds"] for r in rows))
        with self.assertRaises(ValueError):
            b.combine_iteration([rows[0], rows[0]], 0.1)
        rows[1]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            b.combine_iteration(rows, 0.1)

    def test_default_run_is_unprofiled_with_exact_counts_and_incremental_persistence(self):
        case, rpc, saved = self.run_fixture()
        self.assertEqual(case["status"], "completed")
        self.assertTrue(case["eligible_for_unprofiled_timing"])
        self.assertEqual(rpc.invocations, 16)
        self.assertFalse(any(name == "worker_capture" for name, _, _ in rpc.calls))
        measured_counts = [len(c["attempts"][0]["measurements"]) for c in saved if c["attempts"]]
        self.assertTrue(set(range(11)) <= set(measured_counts))
        summary = case["attempts"][0]["summary"]
        self.assertEqual(summary["two_rank_completion_seconds"]["count"], 10)
        self.assertEqual(summary["two_rank_completion_seconds"]["sample_stddev"], 0)

    def test_measured_jit_preserves_all_ten_and_retries_entire_set(self):
        case, rpc, _ = self.run_fixture(FakeNativeRPC(compile_at=("measured", 3, 0)))
        self.assertEqual(case["status"], "completed")
        self.assertEqual(case["accepted_attempt"], 1)
        self.assertEqual(rpc.invocations, 31)
        self.assertEqual([len(a["measurements"]) for a in case["attempts"]], [10, 10])
        self.assertFalse(case["attempts"][0]["summary"]["accepted"])
        self.assertTrue(case["attempts"][0]["measurements"][3]["compilation_observed"])
        self.assertEqual([len(a["warmups"]) for a in case["attempts"]], [5, 5])

    def test_warmup_jit_is_retained_then_five_clean_warmups_follow(self):
        case, rpc, _ = self.run_fixture(FakeNativeRPC(compile_at=("warmup", 4, 0)))
        attempt = case["attempts"][0]
        self.assertEqual(len(attempt["warmup_preparation_rounds"][0]), 5)
        self.assertEqual(len(attempt["warmups"]), 5)
        self.assertEqual(rpc.invocations, 21)
        self.assertTrue(attempt["summary"]["warmup_compilation_free"])

    def test_profile_child_has_only_profile_measurements_and_excludes_warmups(self):
        case, rpc, _ = self.run_fixture(profile=True)
        self.assertEqual(case["status"], "completed")
        self.assertFalse(case["eligible_for_unprofiled_timing"])
        self.assertEqual([args[0] for name, args, _ in rpc.calls if name == "worker_capture"], [True, False])
        iterations = [(args[0], active) for name, args, active in rpc.calls if name == "worker_iteration"]
        self.assertTrue(all(active == (stage == "measured") for stage, active in iterations))
        self.assertEqual(sum(stage == "measured" for stage, _ in iterations), 10)

    def test_failure_retains_prior_measurements_and_stops_capture(self):
        rpc = FakeNativeRPC(fail_at=("measured", 3, 0))
        saved = []
        with self.assertRaisesRegex(RuntimeError, "native rank failed"):
            b.run_case(rpc, "prefill", 256, list(range(4097)), True,
                       lambda c: saved.append(copy.deepcopy(c)))
        self.assertEqual(saved[-1]["status"], "failed")
        self.assertEqual(len(saved[-1]["attempts"][0]["measurements"]), 3)
        self.assertFalse(rpc.capture)
        self.assertFalse(saved[-1]["eligible_for_unprofiled_timing"])

    def test_numerical_failure_never_eligible_and_does_not_discard_observation(self):
        case, rpc, _ = self.run_fixture(FakeNativeRPC(mismatch_at=("measured", 7, 0)))
        self.assertEqual(case["status"], "incomplete")
        self.assertFalse(case["eligible_for_unprofiled_timing"])
        self.assertEqual(len(case["attempts"][0]["measurements"]), 10)
        self.assertEqual(len(case["attempts"]), 1)

    def test_missing_state_check_rejects_completion(self):
        rows = FakeNativeRPC()(b.worker_evidence)
        with self.assertRaisesRegex(ValueError, "missing exact state checks"):
            b.validate_final_state(rows, 16)

    def test_deterministic_prefix_input_policy(self):
        ids, policy = b.input_sequence(336)
        same, _ = b.input_sequence(336)
        other, _ = b.input_sequence(337)
        self.assertEqual(ids, same)
        self.assertNotEqual(ids, other)
        self.assertEqual(len(ids), 4097)
        self.assertTrue(all(0 <= token < b.VOCABULARY for token in ids))
        self.assertEqual(policy["prefix_sha256"]["1024"], b.hash_ids(ids[:1024]))


if __name__ == "__main__":
    unittest.main()
