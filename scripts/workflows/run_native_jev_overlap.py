"""Use permanently released GPUs during the final JEV evaluation tail.

Only NEW scheduler code: native scoring, comparison, manifests, checkpoints,
original adapter scheduler and original native worker remain unchanged.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.jev_pair_protocol import sha256, write_json
from scripts.workflows.native_overlap_gate import read_state, reservations, gpu_inventory, eligible_gpus

ORIGINAL = ROOT / 'scripts/workflows/run_native_jev_comparison.py'
REVIEWED_ADAPTER_SCHEDULER_SHA256 = '089b1a7f1662591626a0d948bbb7dca53012b15a158a37cced8dea7b40e30921'
REVIEWED_NATIVE_WORKER_SHA256 = '1bb16ab567a258d95a0691bbe329809435b7ef644b530535ab10a4724f5403ea'


def make_plan(config_path, output):
    config = json.loads(config_path.read_text())
    if config['protocol'] != 'full_cross_backbone_matched_adapter_study':
        raise ValueError('Wrong study protocol')
    study = ROOT / config['output']
    adapter_plan = json.loads((study / 'parallel_execution_plan.json').read_text())
    if adapter_plan['config'] != config:
        raise ValueError('Adapter config changed')
    scheduler = ROOT / 'scripts/workflows/run_adapter_study_parallel.py'
    if (sha256(scheduler) != REVIEWED_ADAPTER_SCHEDULER_SHA256 or
            adapter_plan['execution']['code_sha256'] != REVIEWED_ADAPTER_SCHEDULER_SHA256):
        raise ValueError('Original scheduler version not reviewed for queue-drained overlap; do not replace a running scheduler')
    if sha256(ORIGINAL) != REVIEWED_NATIVE_WORKER_SHA256:
        raise ValueError('Native worker version differs from the verified 20261008 addition')
    datasets, gaps = {}, {}
    for name in config['datasets']:
        path = ROOT / config['evaluation_root'] / name / 'manifest.json'
        m = json.loads(path.read_text()) if path.exists() else {}
        if m.get('status') != 'ready' or not m.get('complete_split'):
            gaps[name] = m.get('reason', 'No complete public labeled split')
        else:
            datasets[name] = m['N']
    if not datasets or datasets != adapter_plan['datasets']:
        raise ValueError('Full native dataset matrix differs from the running JEV plan')
    print('Checking fixed native model file hashes (CPU/storage only); existing training is unchanged.', flush=True)
    from scripts.evaluation.run_full_before_after import model_identity
    models = {b: model_identity(ROOT / config['visual_models'][b]) for b in config['backbones']}
    if models != adapter_plan['train_identity']['visual_models']:
        raise ValueError('Native model bytes differ from the trained study')
    return config, study, dict(config=config, config_sha256=sha256(config_path), output=str(output),
                               models=models, datasets=datasets, access_gaps=gaps,
                               code_sha256=sha256(ORIGINAL),
                               role='Released native model vs existing trained adapter+same fixed JEV; no new training')


def check_conflicts(state, expected, age, active, admitted):
    reserved, _, _ = reservations(state, expected, age)
    if not active:
        return
    if not state.get('full_accessible_matrix_complete') and state.get('phase') != 'evaluate':
        raise RuntimeError('Original workflow changed stage: stopping only native workers to avoid GPU conflict')
    if reserved.intersection(active):
        raise RuntimeError('Original workflow reclaimed a native GPU: stopping only native workers')
    if (state.get('phase') == 'evaluate' and
            state.get('completed_jobs', 0) < admitted):
        raise RuntimeError('Original workflow appears restarted: stopping only native workers')


def schedule(config, plan, args, output, study):
    queue = [(b, d) for b in config['backbones'] for d in plan['datasets']]
    total = len(queue)
    active, completed, admitted = {}, 0, 0
    logs = ROOT / 'logs/native-vs-jev'
    logs.mkdir(parents=True, exist_ok=True)
    last = 0
    status_path = study / 'pipeline_status.json'
    try:
        while queue or active:
            state, age = read_state(status_path)
            check_conflicts(state, total, age, active, admitted)
            for gpu, (process, handle, b, d, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[gpu]
                if code:
                    raise RuntimeError(f'Native {b}/{d} failed (exit {code}); inspect {log}. Full split remains pending.')
                completed += 1
                print(f'Native completed {completed}/{total}: {b}/{d}', flush=True)
            free, reason = [], 'No pending native jobs'
            if queue:
                devices = gpu_inventory()
                # Read reservations AGAIN after querying actual physical processes.
                state, age = read_state(status_path)
                check_conflicts(state, total, age, active, admitted)
                free, reason = eligible_gpus(args.gpus, state, total, age, devices)
                for gpu in free:
                    if gpu in active or not queue:
                        continue
                    b, d = queue.pop(0)
                    log = logs / f'native_{b}_{d}.log'
                    handle = log.open('a', buffering=1)
                    cmd = [sys.executable, '-u', str(ORIGINAL), '--worker', '--plan', str(output / 'execution_plan.json'),
                           '--backbone', b, '--dataset', d, '--device', f'cuda:{gpu}',
                           '--native-batch', str(args.native_batch), '--cpu-threads', str(args.cpu_threads)]
                    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                           'TOKENIZERS_PARALLELISM': 'false', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                           'OMP_NUM_THREADS': str(args.cpu_threads), 'MKL_NUM_THREADS': str(args.cpu_threads),
                           'OPENBLAS_NUM_THREADS': str(args.cpu_threads), 'TZ': 'Asia/Shanghai'}
                    try:
                        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                    except BaseException:
                        handle.close()
                        raise
                    active[gpu] = (process, handle, b, d, log)
                    admitted = max(admitted, state.get('completed_jobs', 0))
                    print(f'GPU{gpu} released by JEV -> native {b}/{d}, full N={plan["datasets"][d]:,}; log={log}', flush=True)
            if time.monotonic() - last >= 10:
                from scripts.workflows.progress_eta import duration
                progress = {}
                for gpu, (_, _, b, d, _) in active.items():
                    path = output / 'native' / b / d / 'status.json'
                    if path.exists():
                        value = json.loads(path.read_text())
                        progress[str(gpu)] = dict(backbone=b, dataset=d, **value)
                        print(f'GPU{gpu} native {b}/{d}: {value.get("N_complete", 0):,}/{value.get("N_total", 0):,}; '
                              f'task ETA {duration(value.get("task_eta_seconds"))}', flush=True)
                print(f'Native {completed}/{total} complete; native GPUs={sorted(active)}; '
                      f'JEV reservations={sorted(map(int, state.get("active", {})))}; {reason}', flush=True)
                write_json(output / 'pipeline_status.json', dict(status='running', phase='idle_gpu_native_overlap',
                           completed_jobs=completed, total_jobs=total, active_gpus=sorted(active), progress=progress,
                           admission_reason=reason, adapter_state=state))
                last = time.monotonic()
            time.sleep(2)
    finally:
        # Terminate ONLY children started here. Original JEV processes are never signalled.
        for process, _, _, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, _, _, _ in active.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/adapter_study_20261008.json')
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--native-batch', type=int, default=32)
    p.add_argument('--cpu-threads', type=int, default=2)
    args = p.parse_args()
    if (min(args.native_batch, args.cpu_threads) < 1 or len(set(args.gpus)) != len(args.gpus)
            or not args.gpus or min(args.gpus) < 0):
        raise ValueError('Positive batch/thread counts and distinct nonnegative GPU IDs required')
    if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise ValueError('Run in an unmasked all-GPU terminal: CUDA_VISIBLE_DEVICES must be unset')
    # Native parent stays CPU-only even while hashing models and checking results.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output = ROOT / 'experiments/results/native_vs_jev_20261008'
    output.mkdir(parents=True, exist_ok=True)
    from scripts.workflows.adapter_study_acceleration import file_lock
    try:
        with file_lock(output / 'native.lock', nonblocking=True):
            write_json(output / 'pipeline_status.json', dict(status='running', phase='checking_native_model_identity',
                       gpu_workers_started=0, scheduler='idle_gpu_overlap'))
            config, study, plan = make_plan(args.config, output)
            plan['native_batch'] = args.native_batch
            previous = output / 'execution_plan.json'
            if previous.exists() and json.loads(previous.read_text()) != plan:
                raise ValueError('Native scientific plan changed; existing results preserved')
            write_json(previous, plan)
            write_json(output / 'overlap_execution_context.json', dict(
                scheduler_sha256=sha256(__file__), gate_sha256=sha256(ROOT / 'scripts/workflows/native_overlap_gate.py'),
                gpus=args.gpus, cpu_threads=args.cpu_threads, mode='final_jev_queue_drained_and_physical_gpu_free',
                original_scoring_code_unchanged=True,
                latency_scope='Concurrent throughput run; not isolated latency benchmarking'))
            # Children need the unmasked GPU inventory; the parent never calls CUDA.
            del os.environ['CUDA_VISIBLE_DEVICES']
            schedule(config, plan, args, output, study)
            from scripts.workflows.run_native_jev_comparison import acquire_adapter_lock
            from scripts.evaluation.native_jev_comparison import compare_case, build_tables
            # All native GPU workers have exited. Wait for the original final
            # evidence audit and take its lock ONLY for the CPU paired table stage.
            with (study / 'study.lock').open('a+') as lock:
                acquire_adapter_lock(config, lock, output / 'pipeline_status.json')
                for b in config['backbones']:
                    for d in plan['datasets']:
                        print(f'CPU paired evidence audit: {b}/{d}', flush=True)
                        write_json(output / 'pipeline_status.json', dict(status='running', phase='paired_evidence_audit',
                                   backbone=b, dataset=d, gpu_workers_started=0))
                        compare_case(ROOT, config, b, d, output / 'native', study / 'evaluation', output / 'comparisons')
                summary = build_tables(output / 'comparisons', ROOT / 'reports/native_vs_jev_20261008')
                expected = len(config['backbones']) * len(plan['datasets'])
                if summary['complete_cases'] != expected:
                    raise ValueError('Full native/JEV paired comparison count differs')
                gaps = plan['access_gaps']
                final = dict(status='accessible_comparisons_complete_access_gaps_remain' if gaps else 'complete',
                             full_accessible_comparisons_complete=True, complete_cases=expected, expected_cases=expected,
                             pending=[], resource_gaps=gaps, reports=str(ROOT / 'reports/native_vs_jev_20261008'))
                write_json(output / 'pipeline_status.json', final)
                print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
                if gaps:
                    raise SystemExit(2)
    except BlockingIOError:
        raise SystemExit('Native controller already running: Ctrl+C ONLY the waiting native terminal first')
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code == 2:
            raise
        write_json(output / 'pipeline_status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   reason=str(error), artifacts_preserved=True))
        raise


if __name__ == '__main__':
    main()
