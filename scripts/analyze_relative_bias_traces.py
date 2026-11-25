"""Read-only helper attribution through NVTX + CUDA launch correlation.

CPU NVTX ranges enclose launch APIs, not necessarily asynchronous GPU work.
Process and thread identity disambiguate repeated correlation IDs. The outer
forward range synchronizes completion; kernel unions never double-count overlap.
Reference: https://docs.nvidia.com/nsight-systems/AnalysisGuide/
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import hashlib
from pathlib import Path
import re
import sqlite3
import statistics

FULL = re.compile(r"relative_bias/(prefill|decode)/L(256|1024|4096)/(native|fused)/measured/iteration-(\d+)/attempt-(\d+)/block-(\d+)\Z")
HELPER = re.compile(r"relative_bias_helper/(native|fused)/(prefill|decode)/L(256|1024|4096)/measured/iteration-(\d+)/attempt-(\d+)/block-(\d+)/call-(\d+)\Z")


def pid(value):
    return (int(value)//0x1000000)%0x1000000


def union_ns(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end<start:raise ValueError('Negative kernel interval')
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return sum(end-start for start,end in result)


def key(match,helper=False):
    values=match.groups()
    if helper:
        variant,phase,length,iteration,attempt,block,_=values
    else:
        phase,length,variant,iteration,attempt,block=values
    return phase,int(length),variant,int(iteration),int(attempt),int(block)


def safe_symbol(value):
    value=re.sub(r'[^\x20-\x7e]','?',value or 'unknown_kernel')
    return value if len(value)<=320 else value[:290]+'...#'+hashlib.sha256(value.encode()).hexdigest()[:16]


def trace(path):
    """Return every measured range; caller chooses accepted attempts explicitly."""
    path=Path(path)
    if path.is_symlink():raise ValueError('Symlinked trace input is refused')
    full=[];helpers=[];by_thread=defaultdict(list);by_process=defaultdict(list)
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_RUNTIME'}.issubset(tables):
            raise ValueError('CUDA runtime/kernel and NVTX tables required')
        query='SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL'
        for start,end,tid,label in db.execute(query):
            if not isinstance(label,str):continue
            outer_match,helper_match=FULL.fullmatch(label),HELPER.fullmatch(label)
            if not outer_match and not helper_match:continue
            if not isinstance(start,int) or not isinstance(end,int) or end<=start:raise ValueError('Malformed measured range')
            record={'start':start,'end':end,'tid':int(tid),'pid':pid(tid),'label':label,
                    'key':key(outer_match or helper_match,helper=helper_match is not None),'kernels':[], 'helpers':[]}
            if helper_match:
                record['call']=int(helper_match.groups()[-1]);helpers.append(record);by_thread[int(tid)].append(record)
            else:
                full.append(record);by_process[record['pid']].append(record)
        parent_map={}
        for row in full:
            identity=(row['pid'],row['key'])
            if identity in parent_map:raise ValueError('Duplicate same-process measured forward range')
            parent_map[identity]=row
        for helper in helpers:
            parent=parent_map.get((helper['pid'],helper['key']))
            if (parent is None or helper['tid']!=parent['tid'] or
                not parent['start']<=helper['start']<=helper['end']<=parent['end']):
                raise ValueError('Helper range has no unique enclosing same-thread forward')
            parent['helpers'].append(helper)
        for row in full:
            calls=[h['call'] for h in row['helpers']]
            if sorted(calls)!=list(range(1,len(calls)+1)):
                raise ValueError('Missing or duplicate helper call sequence inside a forward')
        helper_starts={};forward_starts={}
        for mapping,starts in ((by_thread,helper_starts),(by_process,forward_starts)):
            for identity,rows in mapping.items():
                rows.sort(key=lambda x:x['start'])
                if any(a['end']>b['start'] for a,b in zip(rows,rows[1:])):raise ValueError('Overlapping ranges cannot be uniquely attributed')
                starts[identity]=[x['start'] for x in rows]
        launches={};all_apis=defaultdict(list)
        # Driver API is optional. A launch may appear in runtime and driver
        # records; matching the same helper is deduplicated, never double-counted.
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for start,end,tid,correlation in db.execute(f'SELECT start,end,globalTid,correlationId FROM {table}'):
                if pid(tid) not in by_process:continue
                if not isinstance(start,int) or not isinstance(end,int) or end<start:
                    raise ValueError('Malformed CUDA API timestamp')
                all_apis[(pid(tid),correlation)].append((start,end,int(tid)))
                if tid not in by_thread:continue
                index=bisect_right(helper_starts[tid],start)-1
                if index<0:continue
                helper=by_thread[tid][index]
                if not helper['start']<=start<=end<=helper['end']:continue
                identity=(pid(tid),correlation)
                if identity in launches and launches[identity] is not helper:raise ValueError('Ambiguous launch correlation')
                launches[identity]=helper
        unmatched_helper_kernels=0
        query='SELECT k.start,k.end,k.deviceId,k.globalPid,k.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start'
        for start,end,device,global_pid,correlation,name in db.execute(query):
            process=pid(global_pid)
            if process not in by_process:continue
            index=bisect_right(forward_starts[process],start)-1
            helper=launches.get((process,correlation))
            if index<0:
                if helper:unmatched_helper_kernels+=1
                continue
            outer=by_process[process][index]
            if not outer['start']<=start<=end<=outer['end']:
                if helper:unmatched_helper_kernels+=1
                continue
            api_rows=all_apis.get((process,correlation),[])
            if not api_rows or not any(outer['start']<=a<=b<=outer['end'] for a,b,_ in api_rows):
                raise ValueError('Measured kernel lacks a correlated CUDA API inside its forward')
            if not isinstance(start,int) or not isinstance(end,int) or end<start:
                raise ValueError('Malformed measured kernel interval')
            kernel={'start':start,'end':end,'device':int(device),'name':safe_symbol(name)}
            outer['kernels'].append(kernel)
            if helper is not None:
                if helper['key']!=outer['key']:raise ValueError('Helper kernel escaped its synchronized forward')
                helper['kernels'].append(kernel)
        if unmatched_helper_kernels:raise ValueError('Attributed helper kernels outside synchronized measured forward')
    output=[]
    for row in full:
        devices={k['device'] for k in row['kernels']}
        if len(devices)!=1:raise ValueError('Each rank forward must have exactly one GPU with kernel activity')
        device=next(iter(devices));work=row['kernels'];owned=[k for h in row['helpers'] for k in h['kernels']]
        if any(not h['kernels'] for h in row['helpers']):raise ValueError('Helper NVTX range has no correlated kernel')
        if any(k['device']!=device for k in owned):raise ValueError('Helper ran on unexpected GPU')
        phase,length,variant,iteration,attempt,block=row['key']
        symbols=defaultdict(list)
        for k in owned:symbols[k['name']].append((k['end']-k['start'])/1e6)
        union=union_ns([(k['start'],k['end']) for k in work]);elapsed=row['end']-row['start']
        helper_union=union_ns([(k['start'],k['end']) for k in owned])
        output.append({'phase':phase,'length':length,'variant':variant,'iteration':iteration,'attempt':attempt,'block':block,
            'pid':row['pid'],'device':device,'range_wall_ms':elapsed/1e6,'kernel_count':len(work),
            'kernel_union_ms':union/1e6,'kernel_coverage_fraction':union/elapsed,'no_kernel_gap_ms':(elapsed-union)/1e6,
            'helper_calls':len(row['helpers']),'helper_kernel_count':len(owned),
            'all_measured_kernels_launch_correlated':True,
            'helper_cpu_range_total_ms':sum(h['end']-h['start'] for h in row['helpers'])/1e6,
            'helper_kernel_sum_ms':sum(k['end']-k['start'] for k in owned)/1e6,
            'helper_kernel_union_ms':helper_union/1e6,'helper_kernel_union_fraction':helper_union/elapsed,
            'helper_projection_spans_sum_ms':sum(max(k['end'] for k in h['kernels'])-min(k['start'] for k in h['kernels']) for h in row['helpers'])/1e6,
            'helper_kernels':[{'name':name,'count':len(v),'sum_ms':sum(v),'mean_ms':statistics.mean(v)} for name,v in sorted(symbols.items())],
            'attribution':'same-process/thread NVTX-enclosed launch API correlated to CUDA kernel; not symbol-name inference'})
    return output
