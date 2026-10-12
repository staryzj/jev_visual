"""Read-only admission checks for native jobs alongside the final JEV stage.

GPU utilization is deliberately NOT an admission criterion. The old scheduler
must have dispatched its entire final queue; an active CPU-preparing worker
still reserves its GPU. Never use this gate to overlap cache/train stages.
"""
import csv
import io
import json
import os
import subprocess
import time
from pathlib import Path


def read_state(path, attempts=8, retry_delay=.02):
    """Read a current stable snapshot; atomic writer replacement is retryable.

    Use descriptor metadata so stat/read do not silently refer to different
    inodes. Retries never grant admission using a changing or malformed state.
    """
    path = Path(path)
    if attempts < 1 or retry_delay < 0:
        raise ValueError('Invalid status-read retry settings')
    reason = 'Status changed during read'
    for attempt in range(attempts):
        try:
            with path.open(encoding='utf-8') as handle:
                before = os.fstat(handle.fileno())
                state = json.load(handle)
                after = os.fstat(handle.fileno())
            current = path.stat()
            signature = lambda s: (s.st_dev, s.st_ino, s.st_mtime_ns, s.st_size)
            if signature(before) == signature(after) == signature(current):
                if not isinstance(state, dict):
                    raise ValueError('Adapter status must be a JSON object; no GPU admission')
                return state, max(0, time.time()-current.st_mtime)
            reason = 'Adapter status replaced/modified while reading'
        except FileNotFoundError:
            reason = 'Adapter status absent while reading'
        except json.JSONDecodeError:
            reason = 'Adapter status JSON incomplete while reading'
        if attempt+1 < attempts:
            time.sleep(retry_delay)
    if not path.exists():
        return {}, float('inf')
    raise RuntimeError(f'Cannot read a stable adapter status after {attempts} attempts: {reason}; no GPU admission')


def reservations(state, expected_jobs, age, max_age=45):
    """Return (GPU reservations, admission allowed, reason), failing closed."""
    if state.get('status') in {'failed', 'interrupted'}:
        raise RuntimeError('Original JEV workflow stopped; preserve outputs and resume it first')
    if state.get('full_accessible_matrix_complete') is True:
        if state.get('complete_cases') != expected_jobs:
            raise ValueError('Completed JEV matrix count differs from the native plan')
        return set(), True, 'Original full accessible JEV matrix complete'
    if state.get('phase') != 'evaluate' or state.get('status') != 'running':
        return set(), False, 'Original workflow is not in its final evaluate stage'
    active = state.get('active')
    total, completed = state.get('total_jobs'), state.get('completed_jobs')
    if (not isinstance(active, dict) or type(total) is not int or
            type(completed) is not int or total != expected_jobs or completed < 0):
        raise ValueError('Unknown original scheduler status; no GPU admission')
    reserved = set()
    for key, value in active.items():
        gpu = int(key)
        if (gpu < 0 or str(gpu) != key or not isinstance(value, dict) or
                type(value.get('pid')) is not int or value['pid'] <= 0 or
                not isinstance(value.get('job'), list) or len(value['job']) != 2):
            raise ValueError('Malformed original GPU reservation')
        reserved.add(gpu)
    pending = total - completed - len(active)
    if pending < 0:
        raise ValueError('Inconsistent original queue counts')
    if age > max_age:
        return reserved, False, 'Original status is stale; waiting for a fresh reservation snapshot'
    if pending:
        return reserved, False, f'Original JEV still has {pending} undispatched jobs (cards may be reused)'
    return reserved, True, f'Original final queue fully dispatched; {len(active)} JEV GPUs remain reserved'


def parse_inventory(gpu_csv, apps_csv):
    devices = {}
    for row in csv.reader(io.StringIO(gpu_csv)):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 4:
            raise ValueError('Unknown nvidia-smi GPU inventory format')
        index, uuid, memory, bus = (cell.strip() for cell in row)
        index, memory = int(index), int(memory)
        if index in devices or not uuid.startswith('GPU-') or memory < 0:
            raise ValueError('Unknown/duplicate physical GPU identity')
        devices[index] = dict(uuid=uuid, memory_mib=memory, bus=bus, pids=set())
    if not devices:
        raise ValueError('No physical GPU inventory')
    # Child CUDA ordinals explicitly use PCI ordering. Fail rather than dispatch
    # to a different physical card on hosts with unusual nvidia-smi ordering.
    ordered = sorted(devices, key=lambda g: devices[g]['bus'])
    if ordered != list(range(len(devices))):
        raise ValueError('Physical GPU indices do not match PCI order; explicit mapping required')
    by_uuid = {v['uuid']: g for g, v in devices.items()}
    if len(by_uuid) != len(devices):
        raise ValueError('Duplicate GPU UUID')
    for row in csv.reader(io.StringIO(apps_csv)):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 2:
            raise ValueError('Unknown nvidia-smi compute process format')
        uuid, pid = (cell.strip() for cell in row)
        if uuid not in by_uuid or int(pid) <= 0:
            raise ValueError('Unmapped/MIG GPU process; no GPU admission')
        devices[by_uuid[uuid]]['pids'].add(int(pid))
    return devices


def gpu_inventory():
    def query(fields, kind):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}',
                                 '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=10)
        return result.stdout
    return parse_inventory(query('index,uuid,memory.used,pci.bus_id', 'gpu'),
                           query('gpu_uuid,pid', 'compute-apps'))


def eligible_gpus(requested, state, expected_jobs, age, devices, max_idle_mib=1024):
    reserved, allowed, reason = reservations(state, expected_jobs, age)
    if any(g not in devices for g in requested):
        raise ValueError('Requested GPU absent from physical inventory')
    # The old controller uses the same physical IDs on the homogeneous server.
    # Detect any observed worker/physical-GPU mismatch before allowing overlap.
    for key, entry in state.get('active', {}).items():
        observed = {g for g, v in devices.items() if entry['pid'] in v['pids']}
        if observed and observed != {int(key)}:
            raise ValueError('Original CUDA ordinal/physical GPU mismatch; no overlap')
    if not allowed:
        return [], reason
    free = [g for g in requested if g not in reserved and not devices[g]['pids']
            and devices[g]['memory_mib'] <= max_idle_mib]
    return free, reason
