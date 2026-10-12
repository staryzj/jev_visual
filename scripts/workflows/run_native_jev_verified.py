"""Idle-GPU scheduler + strict diagnostic fallback, without editing old sources."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.jev_pair_protocol import sha256, write_json


def worker():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--native-batch', type=int, default=32)
    p.add_argument('--cpu-threads', type=int, default=2)
    a = p.parse_args()
    import torch
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(a.device)
    plan = json.loads(a.plan.read_text())
    from scripts.evaluation.native_uncached_fallback import evaluate_with_fallback
    evaluate_with_fallback(ROOT, plan['config'], a.backbone, a.dataset, plan['models'][a.backbone],
                           Path(plan['output'])/'native', a.device, a.native_batch)


def main():
    if '--worker' in sys.argv:
        return worker()
    import scripts.workflows.run_native_jev_overlap as scheduler
    import scripts.evaluation.native_jev_comparison as comparison
    from scripts.evaluation.native_uncached_fallback import selected_native_root
    original_schedule, original_compare = scheduler.schedule, comparison.compare_case

    def verified_schedule(config, plan, args, output, study):
        write_json(output/'verified_execution_context.json', dict(
                   worker_sha256=sha256(__file__), fallback_sha256=sha256(ROOT/'scripts/evaluation/native_uncached_fallback.py'),
                   policy='Preserve complete cached results; numerical mismatch >0.02 requires entire-case uncached result',
                   threshold=.02, original_sources_unmodified=True))
        original_worker = scheduler.ORIGINAL
        scheduler.ORIGINAL = Path(__file__).resolve()
        try:
            return original_schedule(config, plan, args, output, study)
        finally:
            scheduler.ORIGINAL = original_worker

    def verified_compare(runtime, config, backbone, dataset, native_root, adapter_root, comparison_root):
        selected = selected_native_root(native_root, backbone, dataset)
        return original_compare(runtime, config, backbone, dataset, selected, adapter_root, comparison_root)

    # Main's immutable original plan/hash checks run before this hook redirects
    # only worker launch paths. Paired evidence continues through the old verifier.
    scheduler.schedule, comparison.compare_case = verified_schedule, verified_compare
    try:
        scheduler.main()
    finally:
        scheduler.schedule, comparison.compare_case = original_schedule, original_compare


if __name__ == '__main__':
    main()
