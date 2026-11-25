"""CPU tests for state isolation; GPU model correctness remains a separate gate."""
import unittest

import torch

from experiments.context_sweep.state import (
    CacheRegion, StateCheckpoint, exact_bytes_equal, scalar_fingerprint,
    tensor_fingerprint, zero_regions, logical_byte_view,
)


class StateCheckpointTests(unittest.TestCase):
    def fixture(self):
        backing = torch.arange(5*2*4*9,dtype=torch.float32).reshape(5,2,4,9)
        cache = backing[:,:,:,:6]  # Noncontiguous native-style page padding.
        cache[1,0,0,0] = float("nan")
        blocks = torch.tensor([1,3],dtype=torch.int64)
        slots = torch.tensor([6,13],dtype=torch.int64)
        region = CacheRegion("layer.conv",cache,blocks,"convolution",256,3,slots)
        metadata = {"seq_lens":torch.tensor([257],dtype=torch.int32),
                    "positions":torch.tensor([256],dtype=torch.int64),
                    "layout_probe":torch.arange(16).reshape(4,4)}
        scalar = {"history_tokens":256,"query_tokens":1,"total_sequence_tokens":257}
        checkpoint = StateCheckpoint([region],metadata,scalar)
        return backing,region,metadata,scalar,checkpoint

    def test_strided_cache_restores_nan_bytes_and_preserves_unowned_blocks(self):
        backing,region,metadata,scalar,checkpoint = self.fixture()
        unowned = backing[0].clone()
        initial = tensor_fingerprint(region.read())
        region.tensor.index_fill_(0,region.block_indices,0)
        metadata["positions"].fill_(99)
        checkpoint.restore(scalar)
        self.assertEqual(initial,tensor_fingerprint(region.read()))
        self.assertTrue(torch.isnan(region.tensor[1,0,0,0]))
        self.assertTrue(exact_bytes_equal(backing[0],unowned))
        self.assertEqual(metadata["positions"].item(),256)
        self.assertEqual(checkpoint.evidence()["verified_restore_count"],1)

    def test_write_boundary_allows_current_slots_but_rejects_history_overwrite(self):
        _,region,_,_,checkpoint = self.fixture()
        region.tensor[1,:,2,:] += 1  # Physical slot six, scheduled current token.
        self.assertTrue(checkpoint.verify_write_boundaries())
        region.tensor[3,0,0,1] += 1  # Physical slot twelve, cached history.
        with self.assertRaisesRegex(AssertionError,"overwrote history"):
            checkpoint.verify_write_boundaries()

    def test_scalar_growth_rejected_before_restore(self):
        _,region,_,scalar,checkpoint = self.fixture()
        region.tensor[1,0,2,0] = -7
        scalar["total_sequence_tokens"] += 1
        with self.assertRaisesRegex(AssertionError,"advanced or changed"):
            checkpoint.restore(scalar)
        self.assertEqual(region.tensor[1,0,2,0].item(),-7)

    def test_tensor_stride_drift_rejected_even_with_same_shape(self):
        _,_,metadata,scalar,checkpoint = self.fixture()
        metadata["layout_probe"].t_()
        with self.assertRaisesRegex(AssertionError,"layout changed"):
            checkpoint.restore(scalar)

    def test_block_map_drift_rejected(self):
        _,region,_,scalar,checkpoint = self.fixture()
        region.block_indices[0] = 2
        with self.assertRaisesRegex(AssertionError,"physical block map changed"):
            checkpoint.restore(scalar)

    def test_zero_only_selected_native_blocks(self):
        backing,region,_,_,_ = self.fixture()
        old_unowned = backing[2].clone()
        zero_regions([region])
        self.assertEqual(torch.count_nonzero(region.read()).item(),0)
        self.assertTrue(exact_bytes_equal(backing[2],old_unowned))

    def test_tensor_hash_accepts_zero_dimensional_and_empty_tensors(self):
        self.assertEqual(tensor_fingerprint(torch.tensor(3)),tensor_fingerprint(torch.tensor(3)))
        self.assertEqual(len(tensor_fingerprint(torch.tensor([],dtype=torch.int32))),64)

    def test_singleton_strided_and_broadcast_metadata_bytes(self):
        for dtype in (torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32):
            backing = torch.tensor([9, 257, 11], dtype=dtype)
            views = (backing[1::2], backing[1:2].as_strided((1,), (0,)), backing[1::2][:0])
            for value in views:
                with self.subTest(dtype=dtype, shape=value.shape, stride=value.stride()):
                    expected = torch.empty(value.numel(), dtype=dtype)
                    expected.copy_(value)
                    self.assertEqual(logical_byte_view(value).stride(), (1,))
                    self.assertTrue(exact_bytes_equal(value, expected))
                    self.assertEqual(tensor_fingerprint(value, chunk_bytes=1), tensor_fingerprint(expected))
            scalar = backing[1]
            self.assertTrue(exact_bytes_equal(scalar, torch.tensor(257, dtype=dtype)))

    def test_logical_byte_view_preserves_nan_payload_and_signed_zero(self):
        raw = torch.tensor([0x7FC00001, 7, -0x80000000], dtype=torch.int32)
        values = raw.view(torch.float32)[::2]
        expected = torch.tensor([0x7FC00001, -0x80000000], dtype=torch.int32).view(torch.float32)
        self.assertTrue(exact_bytes_equal(values, expected))
        self.assertEqual(tensor_fingerprint(values, chunk_bytes=1), tensor_fingerprint(expected))
        self.assertFalse(exact_bytes_equal(values[1:], torch.tensor([0.0])))
        self.assertFalse(exact_bytes_equal(values[:1], torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)))

    def test_contiguous_cache_byte_view_keeps_existing_storage(self):
        # Hashing still copies to CPU in bounded chunks, without adding an
        # unconditional allocation of a whole existing cache snapshot.
        cache = torch.arange(48, dtype=torch.float32).reshape(2, 2, 4, 3)
        self.assertEqual(logical_byte_view(cache).data_ptr(), cache.data_ptr())

    def test_checkpoint_restores_singleton_strided_metadata(self):
        _, region, metadata, scalar, _ = self.fixture()
        metadata["seq_lens"] = torch.tensor([-1, 257, -1], dtype=torch.int32)[1::2]
        checkpoint = StateCheckpoint([region], metadata, scalar)
        metadata["seq_lens"].fill_(999)
        checkpoint.restore(scalar)
        self.assertEqual(metadata["seq_lens"].item(), 257)
        self.assertEqual(metadata["seq_lens"].stride(), (2,))

    def test_nonfinite_scalar_metadata_rejected(self):
        with self.assertRaises(TypeError):
            scalar_fingerprint({"position":float("nan")})


if __name__ == "__main__":
    unittest.main()
