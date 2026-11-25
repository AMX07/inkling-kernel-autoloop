"""Byte-exact sequence-state checkpoints, always outside model timing.

The caller derives cache regions from the actual native attention/convolution
block tables. This module does not create substitute attention metadata, infer
cache layout, or restore communication protocol counters. Python/scalar
metadata must remain fixed; any drift is rejected rather than repaired silently.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CacheRegion:
    """Native cache tensor and all physical blocks reachable by this case.

    Include blocks read by the prepared history AND blocks written by the
    invocation. Block numbers must come from its actual native metadata. Copies
    retain the native head/token/channel strides when restored into the owner.
    """

    name: str
    tensor: Any
    block_indices: Any
    kind: str  # attention | convolution
    logical_history_tokens: int
    readable_history_tokens: int
    write_slots: Any = None

    def validate(self):
        import torch
        if not self.name or self.kind not in ("attention", "convolution"):
            raise ValueError("A named native cache region and kind are required")
        if not torch.is_tensor(self.tensor) or self.tensor.ndim != 4 or not self.tensor.numel():
            raise ValueError(f"{self.name}: native cache must be a nonempty four-dimensional tensor")
        ids = self.block_indices
        if not torch.is_tensor(ids) or ids.ndim != 1 or ids.dtype != torch.int64 or not ids.numel():
            raise ValueError(f"{self.name}: explicit int64 physical block indices are required")
        if ids.device != self.tensor.device:
            raise ValueError(f"{self.name}: cache and block indices must share a device")
        if bool((ids < 0).any().item()) or bool((ids >= self.tensor.shape[0]).any().item()):
            raise ValueError(f"{self.name}: physical block index is out of bounds")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError(f"{self.name}: checkpoint physical block indices must be unique")
        if not 0 <= self.readable_history_tokens <= self.logical_history_tokens:
            raise ValueError(f"{self.name}: invalid logical/readable history lengths")
        if self.write_slots is not None:
            slots = self.write_slots
            if (not torch.is_tensor(slots) or slots.dtype != torch.int64 or slots.ndim != 1
                    or slots.device != self.tensor.device or not slots.numel()):
                raise ValueError(f"{self.name}: write slots must be a nonempty native int64 vector")
            if bool((slots < 0).any().item()):
                raise ValueError(f"{self.name}: all measured token writes must have real cache slots")
            if not bool(torch.isin(slots//self.tensor.shape[2],ids).all().item()):
                raise ValueError(f"{self.name}: write slots are outside checkpoint blocks")

    def read(self):
        return self.tensor.index_select(0, self.block_indices)


def scalar_fingerprint(record: Mapping[str, Any]) -> str:
    """Require JSON primitive metadata; never stringify arbitrary objects."""
    def validate(value):
        if value is None or type(value) in (bool, int, str):
            return
        if type(value) is float:
            import math
            if math.isfinite(value):
                return
        if type(value) in (list, tuple):
            for item in value:
                validate(item)
            return
        if type(value) is dict and all(type(key) is str for key in value):
            for item in value.values():
                validate(item)
            return
        raise TypeError("Scalar metadata must be finite JSON values with string object keys")
    validate(dict(record))
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def logical_byte_view(tensor, *, device=None):
    """Flatten logical values into a unit-stride byte view for validation.

    Singleton/empty dimensions can be classified contiguous despite a stride
    other than one. A dtype-changing view still requires that stride to be
    one. Preserve already contiguous cache buffers; only materialize when
    needed, or explicitly requested on another device. Same-dtype copies do
    not perform floating arithmetic or normalize NaN/signed-zero bits.
    """
    import torch
    if not torch.is_tensor(tensor):
        raise TypeError("Expected a tensor")
    if device is not None:
        canonical = torch.empty(tensor.numel(), dtype=tensor.dtype, device=device)
        canonical.view(tensor.shape).copy_(tensor.detach())
    else:
        canonical = tensor.detach().contiguous().reshape(-1)
        if canonical.stride(0) != 1:
            destination = torch.empty(canonical.numel(), dtype=canonical.dtype, device=canonical.device)
            destination.copy_(canonical)
            canonical = destination
    return canonical.view(torch.uint8)


def tensor_fingerprint(tensor, chunk_bytes=8 * 1024 * 1024):
    """Chunked CPU hash of exact tensor bytes; never part of a timed region."""
    raw = logical_byte_view(tensor)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape":list(tensor.shape), "dtype":str(tensor.dtype)}).encode())
    for offset in range(0, raw.numel(), chunk_bytes):
        digest.update(raw[offset:offset+chunk_bytes].cpu().numpy().tobytes())
    return digest.hexdigest()


def exact_bytes_equal(left, right):
    import torch
    return (tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
            and torch.equal(logical_byte_view(left), logical_byte_view(right)))


def _tensor_layout(tensor):
    return (tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype,tensor.device,tensor.data_ptr())


def zero_regions(regions: Sequence[CacheRegion]):
    """Make physically cleared selected blocks, before fresh prefill preparation."""
    for region in regions:
        region.validate()
        region.tensor.index_fill_(0, region.block_indices, 0)


class StateCheckpoint:
    """GPU-resident snapshots of only case-reachable native sequence state.

    Creating a checkpoint and hashing it can be expensive. Retain it for one
    case, restore before every warmup/measured iteration, and release it before
    the next case. The caller resets peak-memory counters AFTER restoration and
    synchronization, and reports snapshot residency separately from the model.
    """

    def __init__(self, regions: Sequence[CacheRegion], metadata_tensors: Mapping[str, Any],
                 scalar_metadata: Mapping[str, Any]):
        import torch
        self.regions = tuple(regions)
        names = [region.name for region in regions]
        if not names or len(set(names)) != len(names):
            raise ValueError("State checkpoint requires uniquely named cache regions")
        self.cache = {}
        self.metadata = {}
        self.metadata_targets = dict(metadata_tensors)
        self.region_indices = {}
        self.cache_layouts = {}
        self.metadata_layouts = {}
        self.write_masks = {}
        self.verified_write_boundaries = 0
        self.restores = 0
        self.verified_restores = 0
        self.scalar_sha256 = scalar_fingerprint(scalar_metadata)
        for region in self.regions:
            region.validate()
            self.cache[region.name] = region.read().detach()
            self.region_indices[region.name] = region.block_indices.clone()
            self.cache_layouts[region.name] = _tensor_layout(region.tensor)
            if region.write_slots is not None:
                matches = region.block_indices[:,None] == region.write_slots[None,:]//region.tensor.shape[2]
                rows,columns = matches.nonzero(as_tuple=True)
                mask = torch.zeros((region.block_indices.numel(),region.tensor.shape[2]),
                                   dtype=torch.bool,device=region.tensor.device)
                mask[rows,region.write_slots[columns]%region.tensor.shape[2]] = True
                self.write_masks[region.name] = mask
        for name, tensor in self.metadata_targets.items():
            if not name or not torch.is_tensor(tensor):
                raise ValueError("Metadata fields must be named actual tensors")
            self.metadata[name] = tensor.detach().clone()
            self.metadata_layouts[name] = _tensor_layout(tensor)
        self.cache_hashes = {name:tensor_fingerprint(value) for name,value in self.cache.items()}
        self.metadata_hashes = {name:tensor_fingerprint(value) for name,value in self.metadata.items()}

    def restore(self, scalar_metadata: Mapping[str, Any], *, verify=True):
        if scalar_fingerprint(scalar_metadata) != self.scalar_sha256:
            raise AssertionError("Sequence scalar metadata advanced or changed")
        for region in self.regions:
            if _tensor_layout(region.tensor) != self.cache_layouts[region.name]:
                raise AssertionError(f"{region.name}: native cache layout or storage changed")
            if not exact_bytes_equal(region.block_indices, self.region_indices[region.name]):
                raise AssertionError(f"{region.name}: physical block map changed")
            region.tensor.index_copy_(0, region.block_indices, self.cache[region.name])
        for name, snapshot in self.metadata.items():
            target = self.metadata_targets[name]
            if _tensor_layout(target) != self.metadata_layouts[name]:
                raise AssertionError(f"{name}: metadata tensor layout changed")
            target.copy_(snapshot)
        self.restores += 1
        if verify:
            self.verify(scalar_metadata)
            self.verified_restores += 1

    def verify(self, scalar_metadata: Mapping[str, Any]):
        if scalar_fingerprint(scalar_metadata) != self.scalar_sha256:
            raise AssertionError("Sequence scalar metadata does not match checkpoint")
        for region in self.regions:
            if not exact_bytes_equal(region.read(), self.cache[region.name]):
                raise AssertionError(f"{region.name}: restored sequence cache differs byte-for-byte")
        self.verify_metadata(scalar_metadata)
        return True

    def verify_metadata(self, scalar_metadata: Mapping[str, Any]):
        """Run after forward too: native model code must not advance the schedule."""
        if scalar_fingerprint(scalar_metadata) != self.scalar_sha256:
            raise AssertionError("Sequence scalar metadata advanced during forward")
        for name, snapshot in self.metadata.items():
            if _tensor_layout(self.metadata_targets[name]) != self.metadata_layouts[name]:
                raise AssertionError(f"{name}: metadata tensor layout or storage changed")
            if not exact_bytes_equal(self.metadata_targets[name], snapshot):
                raise AssertionError(f"{name}: metadata advanced during forward")
        return True

    def verify_write_boundaries(self):
        """Reject any write to history/padding outside this call's native slots.

        This byte-level check happens after synchronization and outside timing.
        It does not require a write to change a value, which can legitimately
        remain equal. All four packed conv streams are covered by the same slot.
        """
        import torch
        if len(self.write_masks) != len(self.regions):
            raise ValueError("Write-boundary checking requires native slots for every region")
        for region in self.regions:
            selected = region.read()
            byte_shape = (*selected.shape[:-1], selected.shape[-1] * selected.element_size())
            current = logical_byte_view(selected).reshape(byte_shape)
            previous = logical_byte_view(self.cache[region.name]).reshape(byte_shape)
            changed_slots = (current != previous).any(dim=-1).any(dim=1)
            if bool((changed_slots & ~self.write_masks[region.name]).any().item()):
                raise AssertionError(f"{region.name}: model overwrote history outside scheduled write slots")
        self.verified_write_boundaries += 1
        return True

    def evidence(self):
        regions = []
        total_bytes = 0
        for region in self.regions:
            snapshot = self.cache[region.name]
            copied_bytes = snapshot.numel() * snapshot.element_size()
            total_bytes += copied_bytes
            regions.append({"name":region.name, "kind":region.kind,
                            "cache_shape":list(region.tensor.shape),
                            "cache_strides":list(region.tensor.stride()),
                            "dtype":str(region.tensor.dtype),
                            "allocated_capacity_bytes":region.tensor.numel()*region.tensor.element_size(),
                            "selected_physical_blocks":region.block_indices.numel(),
                            "selected_block_bytes":copied_bytes,
                            "logical_history_tokens":region.logical_history_tokens,
                            "readable_history_tokens":region.readable_history_tokens,
                            "snapshot_sha256":self.cache_hashes[region.name]})
        metadata_bytes = sum(t.numel()*t.element_size() for t in self.metadata.values())
        return {"mechanism":"Exact native physical-block snapshot; index_copy_ restore and byte equality outside timer",
                "regions":regions, "snapshot_cache_bytes":total_bytes,
                "snapshot_metadata_bytes":metadata_bytes,
                "scalar_metadata_sha256":self.scalar_sha256,
                "metadata_sha256":dict(self.metadata_hashes),
                "restore_count":self.restores, "verified_restore_count":self.verified_restores,
                "verified_write_boundary_count":self.verified_write_boundaries,
                "notes":["Snapshot storage is benchmark overhead and remains resident during timing.",
                         "Selected physical-block bytes include padding and future write slots; they are not logical KV payload.",
                         "Communication workspaces/counters are not sequence state and are not reset."]}


def _clone_native_metadata(value):
    """Detach native builder scratch buffers, preserving its exact metadata type."""
    import dataclasses
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{
            field.name:_clone_native_metadata(getattr(value, field.name))
            for field in dataclasses.fields(value) if field.init})
    if isinstance(value, dict):
        return {key:_clone_native_metadata(item) for key,item in value.items()}
    if isinstance(value, list):
        return [_clone_native_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_native_metadata(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported native metadata value {type(value).__name__}")


def _metadata_record(value, prefix, tensors):
    """Collect actual mutable tensors and independently fingerprint scalar state."""
    import dataclasses
    import numpy as np
    import torch
    if torch.is_tensor(value):
        tensors[prefix] = value
        return {"tensor_shape":list(value.shape), "tensor_dtype":str(value.dtype),
                "tensor_strides":list(value.stride()), "tensor_device":str(value.device),
                "tensor_data_ptr":value.data_ptr()}
    if isinstance(value, np.ndarray):
        return {"numpy_shape":list(value.shape), "numpy_dtype":str(value.dtype),
                "numpy_values":value.tolist()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"metadata_class":type(value).__module__+"."+type(value).__qualname__,
                "fields":{field.name:_metadata_record(getattr(value,field.name),
                                                     prefix+"."+field.name,tensors)
                          for field in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {key:_metadata_record(item,prefix+"."+key,tensors)
                for key,item in sorted(value.items())}
    if isinstance(value, (tuple,list)):
        return [_metadata_record(item,prefix+f".{i}",tensors) for i,item in enumerate(value)]
    if value is None or type(value) in (bool,int,float,str):
        return value
    raise TypeError(f"Unsupported sequence metadata value {type(value).__name__}")


class NativeSequenceState:
    """One fixed native single-request invocation; no scheduler or token growth.

    Construct only on an otherwise idle, initialized rank. The parent reserves
    the worker exclusively, with no scheduler requests, for the sweep. We give
    every KV-cache group DISJOINT manager block IDs because native hybrid groups
    overlay the same backing allocation. A preparation prefill and subsequent
    decode use the same capacity, hence the same deterministic physical mapping.
    """

    def __init__(self, *, runner, input_ids, positions, attn_metadata,
                 slot_mapping, regions, history_tokens, capacity_tokens,
                 group_allocations, common_metadata, backing_capacity_bytes):
        self.runner = runner
        self.input_ids = input_ids
        self.positions = positions
        import torch
        self.is_padding = torch.zeros(input_ids.numel(), dtype=torch.bool, device=input_ids.device)
        self.attn_metadata = attn_metadata
        self.slot_mapping = slot_mapping
        self.regions = tuple(regions)
        self.history_tokens = history_tokens
        self.capacity_tokens = capacity_tokens
        self.group_allocations = group_allocations
        self.common_metadata = common_metadata
        self.backing_capacity_bytes = backing_capacity_bytes

    @property
    def query_tokens(self):
        return self.input_ids.numel()

    def _record(self):
        tensors = {}
        record = _metadata_record({"input_ids":self.input_ids,
                                   "positions":self.positions,
                                   "is_padding":self.is_padding,
                                   "attention":self.attn_metadata,
                                   "slot_mapping":self.slot_mapping,
                                   "common":self.common_metadata}, "sequence", tensors)
        record.update({"history_tokens":self.history_tokens,
                       "query_tokens":self.query_tokens,
                       "total_sequence_tokens":self.history_tokens+self.query_tokens,
                       "capacity_tokens":self.capacity_tokens,
                       "group_allocations":self.group_allocations})
        return record,tensors

    def scalar_metadata(self):
        owners = self.runner.vllm_config.compilation_config.static_forward_context
        for region in self.regions:
            if owners[region.name].kv_cache is not region.tensor:
                raise AssertionError(f"{region.name}: native cache binding changed")
        return self._record()[0]

    def make_checkpoint(self):
        scalars,tensors = self._record()
        return StateCheckpoint(self.regions,tensors,scalars)

    def zero(self):
        if self.history_tokens:
            raise ValueError("Cannot zero a prepared decode history")
        zero_regions(self.regions)

    def evidence(self):
        counts = {"attention":0,"convolution":0}
        attention_per_token = 0
        conv_per_token = 0
        owners = self.runner.vllm_config.compilation_config.static_forward_context
        for region in self.regions:
            counts[region.kind] += 1
            if region.kind == "attention":
                attention_per_token += region.tensor.shape[1]*region.tensor.shape[3]*region.tensor.element_size()
            else:
                # Exclude unused power-of-two packing channels. All four
                # streams' critical three-token tail is native readable state;
                # historical SP hidden rows are not universally replicated.
                widths = sum(width for _,width in owners[region.name].stream_ranges)
                conv_per_token += region.tensor.shape[1]*widths*region.tensor.element_size()
        return {"implementation":"pinned native build_attn_metadata; no dummy/profile metadata",
                "query_tokens":self.query_tokens, "history_tokens":self.history_tokens,
                "total_sequence_tokens":self.history_tokens+self.query_tokens,
                "capacity_tokens":self.capacity_tokens,
                "input_ids_sha256":tensor_fingerprint(self.input_ids),
                "positions_sha256":tensor_fingerprint(self.positions),
                "region_counts":counts, "group_allocations":self.group_allocations,
                "unique_backing_capacity_bytes":self.backing_capacity_bytes,
                "logical_payload":{
                    "attention_valid_history_before_bytes":self.history_tokens*attention_per_token,
                    "attention_valid_history_after_bytes":(self.history_tokens+self.query_tokens)*attention_per_token,
                    "attention_basis":"All native per-rank K/V heads and retained history tokens in this direct full-page reservation; excludes page padding.",
                    "convolution_required_tail_before_bytes":min(self.history_tokens,3)*conv_per_token,
                    "convolution_required_tail_after_bytes":min(self.history_tokens+self.query_tokens,3)*conv_per_token,
                    "convolution_full_populated_history_bytes":None,
                    "convolution_basis":"Guaranteed readable last-three-token state across all four actual stream widths, excluding packed-D padding; not total allocated memory. Full populated history differs with native sequence parallelism and is not inferred."},
                "cache_retention":"Full case history pages reserved; native-produced cache values captured. Sequence-parallel convolution replicates critical tail/prefix rows, not every historical hidden-stream row. Native read windows apply; no scheduler eviction is timed.",
                "block_ownership":"Disjoint manager block IDs across hybrid groups, excluding null block zero"}


def build_native_sequence_state(runner, token_ids, *, history_tokens=0,
                                capacity_tokens=None):
    """Build honest native metadata for one full prefill or one-token decode.

    This is pinned to the inspected Inkling2 native BHNC cache API. Unsupported
    layouts fail explicitly instead of risking a dummy or partial computation.
    All allocations/builders/hash work happen outside the model timer.
    """
    import math
    import torch
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

    config = runner.vllm_config
    if config.parallel_config.tensor_parallel_size != 2:
        raise ValueError("Context sweep requires exactly tensor parallel size two")
    if config.parallel_config.data_parallel_size != 1:
        raise ValueError("Context sweep requires one data-parallel request replica")
    if config.parallel_config.decode_context_parallel_size != 1:
        raise ValueError("Context sweep does not use decode context parallelism")
    ids = torch.as_tensor(token_ids, dtype=torch.int64, device=runner.device).reshape(-1)
    query_tokens = ids.numel()
    if query_tokens < 1 or history_tokens < 0 or (history_tokens and query_tokens != 1):
        raise ValueError("Require a nonempty fresh prefill or exactly one decode token")
    if bool(((ids < 0) | (ids >= 200058)).any().item()):
        raise ValueError("Token IDs must lie in the pinned Inkling2 unpadded text vocabulary")
    total = history_tokens+query_tokens
    capacity_tokens = total if capacity_tokens is None else int(capacity_tokens)
    if capacity_tokens < total:
        raise ValueError("Cache capacity must include the current token")
    if capacity_tokens > config.model_config.max_model_len:
        raise ValueError("Model sequence capacity is insufficient")
    if query_tokens > config.scheduler_config.max_num_batched_tokens:
        raise ValueError("Full prefill exceeds native runner capacity; chunking is forbidden")
    positions = torch.arange(history_tokens,total,dtype=torch.int64,device=runner.device)
    q_cpu = torch.tensor([0,query_tokens],dtype=torch.int32)
    seq_cpu = torch.tensor([total],dtype=torch.int32)
    q_gpu = q_cpu.to(runner.device)
    seq_gpu = seq_cpu.to(runner.device)
    groups = runner.kv_cache_config.kv_cache_groups
    if len(groups) != len(runner.attn_groups) or not groups:
        raise ValueError("Native attention groups do not match KV configuration")
    tables,slot_rows,allocations = [],[],[]
    next_block = 1  # zero is the manager's null block
    for group_id,group in enumerate(groups):
        if group.host_resident or not runner.attn_groups[group_id]:
            raise ValueError("Host-resident or empty attention groups are unsupported")
        block_size = int(group.kv_cache_spec.block_size)
        blocks = math.ceil(capacity_tokens/block_size)
        limit = next_block+blocks
        if limit > runner.kv_cache_config.num_blocks:
            raise ValueError("Insufficient disjoint native physical blocks for full case history")
        table = torch.arange(next_block,limit,dtype=torch.int32,device=runner.device)[None,:]
        # Native slot mappings address manager blocks. The layout guard below
        # rejects virtual kernel splitting unsupported by Inkling's slot writes.
        slots = (positions//block_size+next_block)*block_size+positions%block_size
        tables.append(table)
        slot_rows.append(slots)
        allocations.append({"group_id":group_id,"manager_block_size":block_size,
                            "physical_block_begin":next_block,"physical_block_end_exclusive":limit,
                            "physical_block_count":blocks,"capacity_tokens":blocks*block_size,
                            "layer_names":list(group.layer_names)})
        next_block = limit
    stacked_slots = torch.stack(slot_rows)
    metadata = build_attn_metadata(
        attn_groups=runner.attn_groups,num_reqs=1,num_tokens=query_tokens,
        query_start_loc_gpu=q_gpu,query_start_loc_cpu=q_cpu,max_query_len=query_tokens,
        seq_lens=seq_gpu,max_seq_len=total,block_tables=tables,
        slot_mappings=stacked_slots,kv_cache_config=runner.kv_cache_config,
        seq_lens_cpu_upper_bound=seq_cpu,positions=positions,
        is_prefilling=torch.tensor([history_tokens==0],dtype=torch.bool))
    # Native builders have persistent scratch buffers. Clone once per distinct
    # metadata object so preparing another case cannot overwrite this state.
    clones,stable_metadata = {},{}
    for name,md in metadata.items():
        if id(md) not in clones:
            clones[id(md)] = _clone_native_metadata(md)
        stable_metadata[name] = clones[id(md)]
    metadata = stable_metadata
    owners = config.compilation_config.static_forward_context
    regions,slot_mapping = [],{}
    seen_names = set()
    counts = {"attention":0,"convolution":0,"local":0,"full":0}
    for group_id,group in enumerate(groups):
        for name in group.layer_names:
            if name in seen_names or name not in metadata or name not in owners:
                raise ValueError(f"Missing or duplicated native cache owner {name}")
            seen_names.add(name)
            owner,md = owners[name],metadata[name]
            tensor = owner.kv_cache
            owner_type = type(owner).__name__
            if owner_type == "InklingConvState":
                kind = "convolution"
                read_history = min(history_tokens,int(owner.sliding_window)-1)
                if type(md).__name__ != "InklingSconvMetadata":
                    raise TypeError(f"{name}: missing real short-convolution metadata")
                if not bool((md.seq_idx == 0).all().item()) or not bool((md.query_start == 0).all().item()):
                    raise AssertionError("Single-request convolution metadata is inconsistent")
            elif owner_type == "Inkling2Attention":
                kind = "attention"
                counts["local" if owner.is_local else "full"] += 1
                read_history = min(history_tokens,int(owner.sliding_window)-1) if owner.is_local else history_tokens
                if type(md).__name__ != "Inkling2AttentionMetadata":
                    raise TypeError(f"{name}: missing real native attention metadata")
                if md.num_actual_tokens != query_tokens or md.max_seq_len != total:
                    raise AssertionError("Attention token counts do not match the complete invocation")
                if md.query_lens_cpu.tolist() != [query_tokens] or md.seq_lens_cpu.tolist() != [total]:
                    raise AssertionError("CPU attention planning lengths are inconsistent")
                if md.num_decode_tokens != (query_tokens if history_tokens else 0):
                    raise AssertionError("Native attention phase classification is inconsistent")
            else:
                raise TypeError(f"Unsupported cache owner {owner_type}")
            counts[kind] += 1
            if tensor.ndim != 4 or tensor.shape[2] != allocations[group_id]["manager_block_size"]:
                raise ValueError(f"{name}: native kernel/manager block geometry requires an explicit adapter")
            block_table = md.block_table
            if not exact_bytes_equal(block_table,tables[group_id]):
                raise ValueError(f"{name}: native virtual block mapping requires an explicit adapter")
            if not exact_bytes_equal(md.slot_mapping,slot_rows[group_id]):
                raise AssertionError(f"{name}: native slot mapping differs from complete token positions")
            physical_blocks = torch.unique(block_table.reshape(-1).to(torch.int64))
            region = CacheRegion(name,tensor,physical_blocks,kind,history_tokens,read_history,md.slot_mapping)
            region.validate()
            regions.append(region)
            slot_mapping[name] = md.slot_mapping
    if counts != {"attention":48,"convolution":48,"local":40,"full":8}:
        raise AssertionError(f"Pinned Inkling2 cache owner counts differ: {counts}")
    # All descriptors share a single backing allocation in this pinned runtime.
    capacities = {int(desc.size) for desc in runner.kv_cache_config.kv_cache_tensors}
    if len(capacities) != 1:
        raise AssertionError("Expected one shared native backing allocation")
    return NativeSequenceState(
        runner=runner,input_ids=ids,positions=positions,attn_metadata=metadata,
        slot_mapping=slot_mapping,regions=regions,history_tokens=history_tokens,
        capacity_tokens=capacity_tokens,group_allocations=allocations,
        common_metadata={"query_start_loc_cpu":q_cpu,"query_start_loc":q_gpu,
                         "seq_lens_cpu":seq_cpu,"seq_lens":seq_gpu},
        backing_capacity_bytes=capacities.pop())
