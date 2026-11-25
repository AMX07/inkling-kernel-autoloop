"""Fixed-schema PUBLIC projections. Raw logs, tensors and traces remain local.

Unknown fields are dropped at every level, including known object-valued fields.
This validates publication shape/types, not the analyzer's scientific conclusions.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
import re

TOKEN = re.compile(r"(?:eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{15,}|sk-(?:proj-)?[A-Za-z0-9_-]{40,}|hf_[A-Za-z0-9]{25,}|(?:Bearer|Basic)\s+[A-Za-z0-9+/=_-]{12,})", re.I)
PRIVATE_TEXT = re.compile(r"(?:\b(?:api[_-]?key|password|secret|access[_-]?token|authorization)\b\s*[:=]|https?://[^/\s@]+@|Traceback \(most recent call last\)|/Users/|/home/)", re.I)

def text(value):
    if not isinstance(value, str) or len(value) > 20000:
        raise ValueError("Malformed public text")
    if TOKEN.search(value) or PRIVATE_TEXT.search(value):
        return "[Private diagnostic omitted; inspect the local evidence archive]"
    return value

def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Malformed finite public number")
    return value

def integer(value):
    if type(value) is not int:
        raise ValueError("Malformed public integer")
    return value

def boolean(value):
    if type(value) is not bool:
        raise ValueError("Malformed public boolean")
    return value

def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Malformed public SHA256")
    return value

def relative_file(value):
    if (not isinstance(value, str) or value.startswith('/') or '..' in value.split('/')
            or '\\' in value or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)):
        raise ValueError("Malformed relative artifact name")
    return value

def project(value, schema):
    if value is None:
        return None
    if callable(schema):
        return schema(value)
    if isinstance(schema, list):
        if not isinstance(value, list) or len(value) > 100000:
            raise ValueError("Malformed public list")
        return [project(item, schema[0]) for item in value]
    if not isinstance(value, dict):
        raise ValueError("Malformed public object")
    return {key: project(value[key], sub) for key, sub in schema.items() if key in value}

def fields(names, kind=text):
    return {name: kind for name in names.split()}

STATS = {**fields('count', integer), **fields('mean sample_stddev median min max cv', number)}
PATCH = {**fields('target active_variant scope'), **fields('source_sha256 original_helper_sha256 candidate_source_sha256', digest),
         **fields('installed restored fallback_function_unchanged use_sheared_fa4', boolean)}
TENSOR = {**fields('storage_offset logical_bytes', integer), 'shape': [integer], 'stride': [integer], 'dtype': text, 'sha256': digest, 'values': [integer]}
CONTEXT = {**fields('phase stage variant'), **fields('length iteration attempt block block_index', integer)}
EVENT = {**CONTEXT, **fields('backend event fn_name')}
MEMORY = {'before': fields('allocated_bytes reserved_bytes', integer),
          **fields('after_allocated_bytes after_reserved_bytes peak_allocated_bytes peak_reserved_bytes incremental_peak_allocated_bytes incremental_peak_reserved_bytes', integer)}
EXACT = {**fields('label status dtype'), **fields('passed bitwise_equal all_finite output_contiguous has_nonzero_finite_output inputs_unchanged metadata_updated_in_place native_graph_also_verified input_addresses_unchanged', boolean),
         **fields('num_tokens rel_extent input_storage_offset output_bytes', integer),
         **{name: [integer] for name in 'input_shape input_stride output_shape output_stride query_lengths cache_lengths'.split()},
         'metadata_dtype': [text], 'metadata_stride': [[integer]], 'input_sha256': [digest], 'output_sha256': digest}
CAPTURE = {**fields('capture_id file_sha256', digest), 'file': relative_file,
           **fields('rank file_bytes storage_bytes', integer),
           'tensors': {name: TENSOR for name in ('sheared', 'cache_seqlens', 'cu_seqlens_q')},
           'metadata': {'sheared': TENSOR, 'cache_seqlens': TENSOR, 'cu_seqlens_q': TENSOR, 'num_tokens': integer, 'rel_extent': integer},
           'first_context': CONTEXT, 'representative_only': boolean, 'raw_artifact_local_only': boolean}
CAPTURE_CHECK = {'rank': integer, 'captures': [CAPTURE], 'checks': [{'capture_id': digest, 'result': EXACT}],
                 **fields('applicable passed', boolean), **fields('status reason')}
MICRO_OBSERVATION = {**fields('variant'), **fields('iteration repeats before_allocated_bytes before_reserved_bytes peak_allocated_bytes peak_reserved_bytes incremental_peak_allocated_bytes', integer),
                     **fields('wall_us_per_call cuda_event_us_per_call', number), 'compilation_events': [EVENT]}
MICRO = {**fields('kind status execution_mode scope'), 'rank': integer, **fields('instrumented applicable analysis_verified', boolean),
         'reports': [{'capture_id': digest, 'warmup_rounds': [[{'iteration': integer, 'variant': text, 'compilation_events': [EVENT]}]],
                      'observations': [MICRO_OBSERVATION],
                      'summary': {v: {'wall_us_per_call': STATS, 'cuda_event_us_per_call': STATS} for v in ('native', 'fused')},
                      **fields('eligible_for_helper_timing recorded_eligible_for_helper_timing analysis_verified', boolean)}]}
OBSERVATION = {**fields('block iteration', integer), 'wall_ms': number,
               'ranks': [{**fields('rank worker_pid helper_calls', integer), **fields('wall_ms cuda_event_ms', number), 'memory': MEMORY}]}
CASE = {**fields('case_id phase'), **fields('length verified_invocations_per_rank', integer),
        **fields('instrumented verified eligible_for_timing', boolean), 'reasons': [text],
        'latency_ms': {v: STATS for v in ('native','fused')}, **fields('speedup latency_reduction_percent', number),
        'paired_bootstrap95': {**fields('lower upper confidence', number), **fields('resamples seed', integer), 'method': text},
        'blocks': [{'block': integer, 'accepted_attempt': integer, 'native_ms': STATS, 'fused_ms': STATS, 'speedup': number}],
        'observations': {v: [OBSERVATION] for v in ('native','fused')},
        'helper_microbenchmark': [MICRO], 'helper_microbenchmark_verification': {'verified': boolean, 'errors': [text], 'scope': text},
        'captured_input_correctness': [CAPTURE_CHECK]}
TRACE_METRICS = 'range_wall_ms kernel_count kernel_union_ms kernel_coverage_fraction no_kernel_gap_ms helper_calls helper_kernel_count helper_cpu_range_total_ms helper_kernel_sum_ms helper_kernel_union_ms helper_kernel_union_fraction helper_projection_spans_sum_ms'
TRACE_RANGE = {**fields('phase variant attribution'), **fields('length iteration attempt block pid device', integer),
               **fields(TRACE_METRICS, number), 'all_measured_kernels_launch_correlated': boolean,
               'helper_kernels': [{'name': text, 'count': integer, 'sum_ms': number, 'mean_ms': number}]}
TRACE_CASE = {**fields('case_id phase'), 'length': integer, 'verified': boolean, 'errors': [text],
              'devices': [{'device': integer, 'variant': text, 'ranges': integer,
                           **{name: STATS for name in TRACE_METRICS.split()}, 'raw_ranges': [TRACE_RANGE]}]}
SCOPED_INTEGRITY = {'verified': boolean, 'errors': [text], 'scope': text}
INTEGRITY = {**SCOPED_INTEGRITY, 'checks': integer, 'shared': SCOPED_INTEGRITY,
             'sessions': {p: SCOPED_INTEGRITY for p in ('baseline','profile')}}
SESSION = {**fields('verified instrumented weights_retained_between_variants', boolean), 'errors': [text], 'summary_status': text,
           'engine_start_count': integer, 'startup_seconds': number, 'artifact_integrity': SCOPED_INTEGRITY,
           'workers': [{**fields('rank worker_pid adversarial_eager_cases graph_checks', integer),
                        **fields('device_name torch torch_cuda'), 'helper_patch': PATCH}]}
ALLOCATION_PHASE = {**fields('status instrumentation scope'), 'elapsed_seconds': number,
                    **fields('exit_code remaining_live_process_group_count', integer),
                    **fields('timed_out process_group_cleanup_verified post_phase_gpu_idle_verified', boolean)}
ANALYSIS = {**fields('kind run_id generated_at analysis_scope status cloud_status'),
            **fields('analysis_source_sha256 trace_analyzer_source_sha256', digest),
            'integrity': INTEGRITY,
            'sessions': {p: SESSION for p in ('baseline','profile')},
            **fields('baseline_complete profile_complete', boolean), 'cases': [CASE], 'profile_cases': [TRACE_CASE],
            'trace_files': [{'file': relative_file, 'verified': boolean, 'ranges': integer, 'sha256': digest, 'error': text}],
            'decision': {**fields('primary_repeatable_gain decode_mean_no_regression baseline_timing_gates_passed eager_winner eligible_for_serving_validation serving_validated adopted', boolean), 'reason': text},
            'allocation': {**fields('verified app_context_exited children_overlap_verified_absent', boolean),
                           **fields('elapsed_gpu_seconds estimated_gpu_usd gpu_usd_per_second', number),
                           **fields('billing_note app_context_receipt_scope'), 'phases': {p: ALLOCATION_PHASE for p in ('baseline','profile')}},
            'limitations': [text]}
RECOVERY = {**fields('kind run_id baseline_run_id status scope'),
            **fields('verified link_verified profile_complete separate_allocation same_host_assumed baseline_rerun unprofiled_timing_eligible serving_adoption_eligible', boolean),
            **fields('recovery_contract_sha256 baseline_source_provenance_sha256 recovery_source_provenance_sha256 analysis_source_sha256 trace_analyzer_source_sha256', digest),
            'integrity': INTEGRITY, 'profile_session': SESSION, 'profile_cases': [TRACE_CASE],
            'trace_files': ANALYSIS['trace_files'], 'allocation': ANALYSIS['allocation']}
ANALYSIS['profile_recovery'] = RECOVERY

def project_analysis(raw):
    if not isinstance(raw, dict) or raw.get('kind') != 'relative-bias-analysis-v1' or not re.fullmatch(r'relative-bias-\d{8}T\d{6}Z', str(raw.get('run_id'))):
        raise ValueError('Unsupported relative-bias analysis')
    return project(raw, ANALYSIS)


def link_profile_recovery(baseline, recovery, root):
    """Verify a separate Nsight allocation without completing the original run.

    Recompute the two analyses from immutable evidence. Input analysis files are
    checked for staleness, not trusted as pass flags. Only bounded hash-linked
    provenance and the fixed-schema profile projection can reach the website.
    """
    from pathlib import Path
    from scripts.analyze_relative_bias import analyze
    root=Path(root).resolve()
    baseline=project_analysis(baseline);recovery=project_analysis(recovery)
    if baseline['run_id']==recovery['run_id']:
        raise ValueError('Profile recovery must be a separate immutable run')
    baseline_run=root/'runs'/baseline['run_id'];recovery_run=root/'runs'/recovery['run_id']
    def read(path):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Missing or unsafe recovery evidence')
        return json.loads(path.read_text())
    def sha(path):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):raise ValueError('Missing recovery-linked evidence')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
        return h.hexdigest()
    for run in (baseline_run,recovery_run):
        receipt=read(run/'cloud-evidence.json')
        if (receipt.get('kind')!='relative-bias-experiment' or receipt.get('id')!=run.name
                or receipt.get('status') not in ('finished','failed','incomplete','interrupted','cancelled')
                or not receipt.get('finished_at') or receipt.get('download_error_type')):
            raise ValueError('Recovery publication requires terminal, fully downloaded attempts')
    actual_baseline=project_analysis(analyze(baseline_run));actual_recovery=project_analysis(analyze(recovery_run))
    def stable(value):
        return {k:v for k,v in value.items() if k not in ('generated_at','profile_recovery')}
    if stable(baseline)!=stable(actual_baseline) or stable(recovery)!=stable(actual_recovery):
        raise ValueError('Derived analysis is stale or differs from independently recomputed evidence')
    if actual_baseline.get('baseline_complete') is not True:
        raise ValueError('The linked original baseline is not independently complete')
    provenance=read(recovery_run/'source-provenance.json')
    original_provenance=read(baseline_run/'source-provenance.json')
    link=provenance.get('profile_only_recovery',{})
    name='profile-only-recovery.json';contract_path=recovery_run/'executed-source/contracts'/name
    contract=read(contract_path);contract_hash=sha(contract_path)
    if (provenance.get('kind')!='relative-bias-source-provenance'
            or link.get('contract')!='contracts/'+name or link.get('sha256')!=contract_hash
            or link.get('baseline_run_id')!=baseline['run_id']
            or provenance.get('contract_hashes',{}).get(name)!=contract_hash
            or contract.get('kind')!='verified-relative-bias-profile-recovery-v1'
            or contract.get('baseline_run_id')!=baseline['run_id'] or contract.get('verified') is not True
            or contract.get('separate_allocation') is not True or contract.get('same_host_assumed') is not False
            or contract.get('baseline_rerun') is not False or contract.get('unprofiled_timing_eligible') is not False
            or contract.get('serving_adoption_eligible') is not False):
        raise ValueError('Invalid immutable profile recovery contract')
    required={'cloud-evidence.json','source-provenance.json','manifest.json','benchmark-baseline/manifest.json',
              'benchmark-baseline/summary.json','benchmark-baseline/runtime.json','benchmark-baseline/inputs.json'}
    linked=contract.get('baseline_artifact_hashes',{})
    if set(linked)!=required or any(sha(baseline_run/name)!=digest(expected) for name,expected in linked.items()):
        raise ValueError('Profile recovery points to different original baseline artifacts')
    from modal_relative_bias import RECOVERY_SOURCE_FILES
    matching=contract.get('matching_experiment_source_hashes',{})
    if (set(matching)!=set(RECOVERY_SOURCE_FILES) or any(
            digest(expected)!=original_provenance.get('source_hashes',{}).get(name)
            or expected!=provenance.get('source_hashes',{}).get(name) for name,expected in matching.items())
            or contract.get('baseline_source_hashes')!=original_provenance.get('baseline_source_hashes')
            or contract.get('baseline_source_hashes')!=provenance.get('baseline_source_hashes')
            or contract.get('plan_sha256')!=original_provenance.get('experiment_plan_sha256')
            or contract.get('plan_sha256')!=provenance.get('experiment_plan_sha256')):
        raise ValueError('Recovered profile source or plan differs from the original baseline')
    receipt=read(recovery_run/'cloud-evidence.json');worker=read(recovery_run/'cloud-worker.json')
    if (receipt.get('profile_only') is not True or receipt.get('requested_phases')!=['profile']
            or worker.get('profile_only') is not True or worker.get('requested_phases')!=['profile']
            or worker.get('unprofiled_timing_performed') is not False
            or set(worker.get('phases',{}))!={'profile'}):
        raise ValueError('Recovered allocation did not follow the profile-only contract')
    integrity=actual_recovery.get('integrity',{});session=actual_recovery.get('sessions',{}).get('profile',{})
    complete=(actual_recovery.get('profile_complete') is True and integrity.get('shared',{}).get('verified') is True
              and integrity.get('sessions',{}).get('profile',{}).get('verified') is True and session.get('verified') is True)
    return project({'kind':'linked-relative-bias-profile-recovery-v1','run_id':recovery['run_id'],
        'baseline_run_id':baseline['run_id'],'status':actual_recovery.get('status'),'verified':complete,'link_verified':True,
        'profile_complete':complete,'separate_allocation':True,'same_host_assumed':False,'baseline_rerun':False,
        'unprofiled_timing_eligible':False,'serving_adoption_eligible':False,'recovery_contract_sha256':contract_hash,
        'baseline_source_provenance_sha256':sha(baseline_run/'source-provenance.json'),
        'recovery_source_provenance_sha256':sha(recovery_run/'source-provenance.json'),
        'analysis_source_sha256':actual_recovery.get('analysis_source_sha256'),
        'trace_analyzer_source_sha256':actual_recovery.get('trace_analyzer_source_sha256'),
        'integrity':integrity,'profile_session':session,'profile_cases':actual_recovery.get('profile_cases',[]),
        'trace_files':actual_recovery.get('trace_files',[]),'allocation':actual_recovery.get('allocation',{}),
        'scope':'Independent Nsight recovery from a separate B300:2 allocation; host and GPU identities may differ. The original baseline run and its adoption decision remain unchanged. These instrumented traces do not supply new unprofiled timings or a cross-allocation profiler-overhead estimate.'},RECOVERY)

def _template(value):
    if isinstance(value, dict): return {k: _template(v) for k,v in value.items()}
    if isinstance(value, list): return [_template(value[0])] if value else [text]
    if value is None: return lambda v: None if v is None else (_ for _ in ()).throw(ValueError('Expected null'))
    return {bool: boolean, int: integer, float: number, str: text}[type(value)]

def project_config(raw, frozen_plan):
    # Exact protocol equality prevents an arbitrary config map masquerading as
    # the public plan. The current plan supplies a bounded *key/type schema*;
    # values remain the selected run's frozen protocol, not current defaults.
    from experiments.relative_bias.benchmark import plan, canonical_hash
    projected_plan = project(frozen_plan, _template(plan()))
    if projected_plan != frozen_plan:
        raise ValueError('Frozen plan contains unexpected/private fields')
    if any(raw.get(k) != v for k,v in frozen_plan.items()) or raw.get('plan_sha256') != canonical_hash(frozen_plan):
        raise ValueError('Recorded config does not match frozen plan')
    return {**projected_plan, 'instrumented': boolean(raw['instrumented']), 'plan_sha256': digest(raw['plan_sha256'])}

def project_inputs(raw, frozen_plan):
    from experiments.context_sweep.benchmark import input_sequence
    _, expected = input_sequence(integer(frozen_plan['seed']))
    if any(raw.get(k) != v for k,v in expected.items()):
        raise ValueError('Input IDs/prefix hashes differ from frozen seeded policy')
    return expected

def project_inventory(raw):
    from experiments.context_sweep.inventory import SAFE_ENVIRONMENT
    schema = {**fields('kind phase captured_at scope python platform os_release torch_build_config'),
              'schema_version': integer, 'gpu_requested': boolean,
              'distributions': [{'name': text, 'version': text}],
              'active_runtime': fields('torch torch_cuda triton triton_module'),
              'torch_cuda_context_initialized_in_inventory_process': boolean,
              'selected_environment': fields(' '.join(SAFE_ENVIRONMENT))}
    result = project(raw, schema)
    # Never republish raw command stdout/stderr or arbitrary patch documents.
    # Hashes keep the exact local records identifiable without exposing them.
    for key in ('commands','compatibility_patches'):
        if key in raw:
            payload=json.dumps(raw[key],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            result[key+'_sha256']=hashlib.sha256(payload).hexdigest()
    versions={}
    patterns={'nsys': r'Nsight Systems\s+(\d+(?:\.\d+){1,4})',
              'nvcc': r'\brelease\s+(\d+(?:\.\d+){1,3})',
              'uv': r'^uv\s+(\d+(?:\.\d+){1,3})'}
    for command in raw.get('commands',[]):
        argv=command.get('argv') if isinstance(command,dict) else None
        if argv in ([name,'--version'] for name in patterns) and command.get('exit_code')==0:
            match=re.search(patterns[argv[0]],str(command.get('stdout','')),re.M)
            if match: versions[argv[0]]=match.group(1)
    result['tool_versions']=versions
    result['publication_note']='Raw command output and patch records remain in the local archive; their canonical JSON hashes are provided here.'
    return result
