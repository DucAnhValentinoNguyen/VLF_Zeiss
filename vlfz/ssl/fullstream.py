"""Sequential pilots, full runs, evaluation and artifact publication.

Receipts make restarts idempotent. Only a checkpointed wall-time exit (75)
requests resubmission; all other failures require inspection.
"""
import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..data.portal import atomic_json


def preflight(root):
    import re
    import torch
    # A fresh Slurm GPU allocation can briefly report CUDA unavailable while
    # NVML initializes. Do not discard an otherwise valid 24-hour allocation
    # for that transient condition.
    for attempt in range(1, 7):
        if torch.cuda.is_available():
            break
        if attempt == 6:
            raise RuntimeError('GPU unavailable after readiness retries')
        print(f'[fullstream] GPU not ready (attempt {attempt}/6); retrying in 15s', flush=True)
        time.sleep(15)
    if root.is_symlink() or root.resolve().parent != Path.home().resolve():
        raise RuntimeError('Fullstream root must be a real directory under home')
    q = subprocess.run(['/usr/lpp/mmfs/bin/mmlsquota', '-u', os.environ['USER'], 'dsshome1'],
                       capture_output=True, text=True, timeout=60, check=True)
    line = next((s for s in q.stdout.splitlines() if re.match(r'dsshome1\s+USR\s', s)), None)
    if not line:
        raise RuntimeError('Cannot validate home quota')
    fields = line.split()
    used, soft = int(fields[2]), int(fields[3])
    cache = Path(os.environ['PORTAL_CACHE'])
    cached = sum(p.stat().st_size for p in cache.glob('*.zip'))
    needed = max(0, 16*2**30 - cached) + 8*2**30
    if (soft-used)*1024 < needed:
        raise RuntimeError('Insufficient quota for cache growth and atomic checkpoints')
    root.mkdir(exist_ok=True)
    probe = root / '.checkpoint-write-test'
    with probe.open('wb') as f:
        f.write(bytes(1024*1024))
        f.flush()
        os.fsync(f.fileno())
    probe.unlink()


def main():
    root = Path(os.environ['OUT_ROOT']).expanduser()
    preflight(root)
    objectives = os.environ.get('STREAM_OBJECTIVES', 'dino,lejepa').split(',')
    if not objectives or any(x not in ('dino', 'lejepa') for x in objectives):
        raise ValueError('STREAM_OBJECTIVES must contain dino and/or lejepa')
    locks = []
    for objective in sorted(set(objectives)):
        lock = (root / f'.{objective}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    child = None

    def forward(*_):
        if child is not None:
            child.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, forward)

    def evaluate(objective):
        # Evaluation shares feature caches and report files across objectives.
        with (root / '.evaluation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            evaluate_locked(objective)

    def evaluate_locked(objective):
        receipt = root / f'evaluation_{objective}.json'
        if receipt.exists():
            return
        for stage in ('pre', 'post'):
            args = ['--init', 'imagenet', '--objective', objective, '--stage', stage,
                    '--run-tag', 's506_raw_e3' if stage == 'post' else '']
            subprocess.run([sys.executable, '-m', 'vlfz.eval.run_eval', *args], check=True)
            subprocess.run([sys.executable, '-m', 'vlfz.eval.seg', *args], check=True)
        atomic_json(receipt, {'complete': True})
        subprocess.run([sys.executable, '-m', 'vlfz.report.aggregate'], check=True)

    for phase in ('pilot', 'full'):
        for objective in objectives:
            receipt = root / f'{phase}_{objective}.json'
            if receipt.exists():
                if phase == 'full':
                    evaluate(objective)
                continue
            tag = 's506_raw_e3' + ('_pilot' if phase == 'pilot' else '')
            env = dict(os.environ, OBJ=objective, RUN_TAG=tag)
            pilot = root / f'pilot_{objective}.json'
            if phase == 'full':
                env['GRAD_CKPT'] = str(json.loads(pilot.read_text())['grad_checkpointing']).lower()
            args = [sys.executable, '-m', 'vlfz.ssl.pretrain', '--source', 'portal_zip',
                    '--init', 'imagenet', '--objective', objective, '--stage', 'full', '--run-tag', tag]
            if phase == 'pilot':
                args += ['--limit-steps', '200']
            started = time.time()
            log = root / f'{phase}_{objective}.log'
            with log.open('a') as f:
                child = subprocess.Popen(args, env=env, stdout=f, stderr=subprocess.STDOUT)
                rc = child.wait()
            child = None
            if rc and rc != 75 and phase == 'pilot' and 'CUDA out of memory' in log.read_text():
                env['GRAD_CKPT'] = 'true'
                with log.open('a') as f:
                    child = subprocess.Popen(args, env=env, stdout=f, stderr=subprocess.STDOUT)
                    rc = child.wait()
                child = None
            if rc:
                # A failed pilot never becomes a full run automatically.
                raise SystemExit(rc)
            atomic_json(receipt, {'seconds': time.time()-started, 'run_tag': tag,
                                  'throughput': json.loads((root / 'ssl' / f'{objective}_imagenet_gastronet_full__{tag}' / 'throughput.json').read_text()),
                                  'grad_checkpointing': env.get('GRAD_CKPT', 'false') == 'true'})
            if phase == 'full':
                evaluate(objective)
    import wandb
    with wandb.init(project=os.environ.get('WANDB_PROJECT', 'vlf-zeiss'), job_type='publish') as run:
        for objective in objectives:
            artifact = wandb.Artifact(f'{objective}-imagenet-s506-raw-e3', type='model')
            artifact.add_file(str(root / 'ssl' / f'{objective}_imagenet_gastronet_full__s506_raw_e3' / 'ema_backbone.pt'))
            run.log_artifact(artifact).wait()


if __name__ == '__main__':
    main()
