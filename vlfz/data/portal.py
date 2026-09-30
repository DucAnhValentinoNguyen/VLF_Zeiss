"""Finite, resumable Cortex ZIP stream with a two-archive disk budget.

Only the training consumer commits progress. Downloads and decoding never
advance the durable cursor. No authentication material enters the manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import requests
from torch.utils.data import IterableDataset

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def _init_transform_worker(transform):
    import torch
    global _worker_transform
    _worker_transform = transform
    torch.set_num_threads(1)


def _transform_sample(raw, identity, transform):
    import torch
    import numpy as np
    # PortalStream.decode already turns a truncated/corrupt image, or a
    # missing archive member (raw=None), into a deterministic blank -- it
    # never raises, so there's nothing to catch here.
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], 'big')
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            return transform(PortalStream.decode(raw))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def _worker_sample(task):
    return _transform_sample(*task, _worker_transform)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class PortalClient:
    def __init__(self, spec_path):
        self.spec_path = Path(spec_path).expanduser()
        self.transferred_bytes = 0
        self.reload()

    def reload(self):
        from pipeline.ingest.portal_to_s3 import _session
        self.spec = json.loads(self.spec_path.read_text())
        self.base = self.spec.get('cortex_base', 'https://cortex.thetavision.nl').rstrip('/')
        self.session = _session(self.spec)

    def renew(self):
        access = os.environ.get('CORTEX_ACCESS_URL')
        if not access:
            self.reload()
            return
        s = self.session
        r = s.get(access, timeout=60)
        if r.status_code != 200:
            raise RuntimeError('Portal login failed; refresh credentials')
        token = r.url.rstrip('/').split('/')[-1]
        b = s.get(self.base + '/api/bootstrap/', timeout=60)
        csrf = b.json()['csrf_token']
        r = s.post(self.base + '/api/request/login/', json={'token': token},
                   headers={'X-Csrftoken': csrf, 'Referer': r.url}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError('Portal login rejected; refresh credentials')
        b = s.get(self.base + '/api/bootstrap/', timeout=60)
        s.headers['x-csrftoken'] = b.json()['csrf_token']

    def url(self, item):
        # 401/403 -> renew the session once, then retry. 5xx or a network
        # exception -> the portal is transiently unreachable, not wrong (a
        # real ~2h HTTP 500 window killed both training jobs on 2026-09-24;
        # download() already retried transient failures, this call never
        # did, and an earlier version of this fix only budgeted ~31s total,
        # which would NOT have survived that outage). Retrying costs nothing
        # -- the GPU allocation is already held either way -- while dying
        # costs however long it takes a human to notice plus re-entering a
        # queue that has taken up to 2 days under real contention, so budget
        # for hours, not seconds. Any OTHER status (404, 400, ...) is still
        # fatal immediately: that's a real, permanent problem, not a
        # transient one, and retrying it forever would just hide it.
        renewed = False
        deadline = time.monotonic() + 3 * 3600  # 3h -- comfortably past the observed 2h outage
        attempt = 0
        while True:
            try:
                r = self.session.post(f"{self.base}/api/provided_file/{item['id']}/download_url/",
                                      data='{}', timeout=60)
            except requests.RequestException as e:
                r = None
                transient = e
            else:
                transient = None
                if r.status_code == 200:
                    return r.json()['url']
                if r.status_code in (401, 403) and not renewed:
                    renewed = True
                    try:
                        self.renew()
                    except requests.RequestException:
                        raise RuntimeError('Portal renewal failed; refresh private credentials') from None
                    continue
                if r.status_code < 500:
                    raise RuntimeError(f'Portal authorization/request failed: HTTP {r.status_code}')
            # Transient (network exception, or a 5xx from the portal itself).
            if time.monotonic() >= deadline:
                detail = f'HTTP {r.status_code}' if r is not None else repr(transient)
                raise RuntimeError(f'Portal still unreachable after 3h of retrying ({detail})')
            wait = min(2 ** attempt, 300)  # ramps to a 5-minute cap, not 30s
            print(f'[portal] url() attempt {attempt+1} failed '
                  f'({"HTTP " + str(r.status_code) if r is not None else repr(transient)}) -- '
                  f'retrying in {wait}s', flush=True)
            time.sleep(wait)
            attempt += 1

    def range(self, item, start, end):
        try:
            response = requests.get(self.url(item), headers={'Range': f'bytes={start}-{end}'},
                                    timeout=(30, 120), stream=True)
        except requests.RequestException:
            raise OSError('Range network failure') from None
        with response as r:
            expected = f"bytes {start}-{end}/{item['size']}"
            if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                raise OSError('Range unavailable')
            data = r.raw.read(end - start + 2)
            self.transferred_bytes += len(data)
            if len(data) != end - start + 1:
                raise OSError('Truncated range')
            return data

    def download(self, item, path):
        expected = item['size']
        for attempt in range(6):
            have = path.stat().st_size if path.exists() else 0
            if have == expected:
                return path
            if have > expected:
                raise RuntimeError('Oversized cached ZIP; inspect cache')
            try:
                with requests.get(self.url(item), headers={'Range': f'bytes={have}-'},
                                  stream=True, timeout=(30, 120)) as r:
                    if r.status_code not in (200, 206):
                        raise OSError('Download rejected')
                    if r.status_code == 206 and r.headers.get('Content-Range') != f'bytes {have}-{expected-1}/{expected}':
                        raise OSError('Invalid Content-Range')
                    mode = 'ab' if r.status_code == 206 else 'wb'
                    count = have if mode == 'ab' else 0
                    with path.open(mode) as f:
                        for block in r.iter_content(1 << 20):
                            if count + len(block) > expected:
                                raise RuntimeError('Server exceeded declared ZIP size')
                            f.write(block)
                            count += len(block)
                            self.transferred_bytes += len(block)
                if count == expected:
                    return path
            except requests.RequestException:
                pass  # Never print requests exceptions: they contain signed URLs.
            except OSError as e:
                if e.errno in (28, 122):
                    raise RuntimeError('Storage full or quota exceeded') from None
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Download exhausted retries: {item['file_name']}")


class RangeReader(io.RawIOBase):
    def __init__(self, client, item):
        self.client, self.item, self.pos = client, item, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.item['size'] if whence == 2 else 0)
        return self.pos

    def read(self, size=-1):
        end = self.item['size'] if size < 0 else min(self.item['size'], self.pos + size)
        if end <= self.pos:
            return b''
        result = self.client.range(self.item, self.pos, end - 1)
        self.pos = end
        return result


class ZipCache:
    def __init__(self, root, budget=16 * 2**30):
        import fcntl
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.budget, self.peak = budget, 0

    def reserve(self, items):
        wanted = {str(x['id']) + '.zip': x['size'] for x in items}
        if sum(wanted.values()) > self.budget:
            raise RuntimeError('ZIP pair exceeds cache budget')
        for p in self.root.glob('*.zip'):
            if p.name not in wanted:
                p.unlink()
        self.peak = max(self.peak, sum(wanted.values()))
        return [self.root / (str(x['id']) + '.zip') for x in items]


def members(z):
    return [i.filename for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(EXTENSIONS)]


def _archive_kind(item_or_path):
    name = str(item_or_path.get('file_name') if isinstance(item_or_path, dict) else item_or_path).lower()
    if name.endswith('.zip'):
        return 'zip'
    if name.endswith('.7z'):
        return '7z'
    raise RuntimeError(f'Unsupported portal archive type: {name}')


def _sevenzip_bin():
    exe = shutil.which('7z') or shutil.which('7zz') or shutil.which('7za')
    if not exe:
        raise RuntimeError('7z executable is required for .7z portal archives')
    return exe


def _run_7z(args):
    try:
        return subprocess.run([_sevenzip_bin(), *args], check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode('utf-8', 'replace').splitlines()[-1:] or ['7z failed']
        raise RuntimeError(msg[0]) from None


def sevenzip_members(path):
    out = _run_7z(['l', '-slt', str(path)]).stdout.decode('utf-8', 'replace').splitlines()
    result = []
    current = {}
    for line in out + ['']:
        if not line:
            name = current.get('Path')
            if name and current.get('Folder') != '+' and name.lower().endswith(EXTENSIONS):
                result.append(name)
            current = {}
            continue
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            current[k] = v
    return result


def archive_members(path, item):
    kind = _archive_kind(item)
    if kind == 'zip':
        with zipfile.ZipFile(path) as z:
            return members(z)
    return sevenzip_members(path)


def archive_read(path, item, member):
    kind = _archive_kind(item)
    if kind == 'zip':
        with zipfile.ZipFile(path) as z:
            return z.read(member)
    return _run_7z(['x', '-so', str(path), member]).stdout


@contextmanager
def archive_reader(path, item):
    """Yield a member reader while parsing a ZIP central directory only once.

    Portal ZIPs contain roughly 10k images each. Reconstructing ``ZipFile`` for
    every image reparses that large directory roughly 10k times per archive and
    dominated the input pipeline. 7z keeps the existing per-member fallback for
    now because the CLI has no equivalent persistent random-access handle.
    """
    if _archive_kind(item) == 'zip':
        with zipfile.ZipFile(path) as z:
            yield z.read
    else:
        # Extract once, retaining random member order without launching a 7z
        # process (and potentially decoding a solid block) for every image.
        listing = _run_7z(['l', '-slt', str(path)]).stdout.decode('utf-8', 'replace')
        records = []
        for block in listing.split('\n\n'):
            fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
            if 'Size' in fields and 'Path' in fields:
                records.append(fields)
        if not records:
            raise RuntimeError('Cannot validate 7z extraction size')
        expanded = 0
        for record in records:
            name = Path(record['Path'])
            if (name.is_absolute() or '..' in name.parts or
                    'Symbolic Link' in record or 'Hard Link' in record or
                    'l' in record.get('Attributes', '')):
                raise RuntimeError('Unsafe 7z member path or link')
            expanded += int(record['Size'])
        cache_root = Path(path).parent
        # Reserve declared sizes, including the concurrently downloading ZIP.
        occupied = sum(p.stat().st_size for p in cache_root.glob('*.zip'))
        # Caller supplies the full current/next reservation via the item copy.
        occupied = max(occupied, item.get('_reserved_bytes', occupied))
        if expanded + occupied > 16 * 2**30:
            raise RuntimeError('7z extraction plus ZIP pair exceeds 16 GiB cache budget')
        with tempfile.TemporaryDirectory(prefix='extract-', dir=cache_root) as directory:
            _run_7z(['x', '-y', '-mmt=2', '-o' + directory, str(path)])
            root = Path(directory).resolve()
            def read(member):
                target = (root / member).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise RuntimeError('Invalid extracted member')
                return target.read_bytes()
            yield read


def index_manifest(client, cache, path, expected=506):
    items = sorted([{k: f[k] for k in ('id', 'file_name', 'size')} for f in client.spec['files']],
                   key=lambda f: f['file_name'])
    if len(items) != expected or len({f['id'] for f in items}) != expected:
        raise RuntimeError(f'Expected {expected} distinct portal archives')
    digest = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    path = Path(path)
    result = json.loads(path.read_text()) if path.exists() else {'fingerprint': digest, 'archives': []}
    if result['fingerprint'] != digest:
        raise RuntimeError('Portal manifest changed; use a new run')
    for item in items[len(result['archives']):]:
        if _archive_kind(item) == 'zip':
            try:
                with zipfile.ZipFile(RangeReader(client, item)) as z:
                    names = members(z)
            except (OSError, zipfile.BadZipFile):
                local = client.download(item, cache.reserve([item])[0])
                names = archive_members(local, item)
        else:
            local = client.download(item, cache.reserve([item])[0])
            names = archive_members(local, item)
        if not names or len(names) != len(set(names)):
            raise RuntimeError('Empty or ambiguous archive image index')
        result['archives'].append({**item, 'members': names})
        atomic_json(path, result)
        print(f"[portal index] {len(result['archives'])}/{expected}", flush=True)
    return result


def batch_sizes(n, size):
    if n < 2 or size < 2:
        raise ValueError('SSL requires at least two samples')
    full, tail = divmod(n, size)
    sizes = [size] * full
    if tail == 1 and sizes:
        sizes[-1] += 1
    elif tail:
        sizes.append(tail)
    return sizes


class PortalStream(IterableDataset):
    """Iterable batch source. Call commit only after a successful optimizer step.

    A corrupt/truncated image, or an archive member the manifest lists but
    the archive no longer has, is replaced with a deterministic blank sample
    rather than dropped or raised -- see PortalStream.decode. That keeps
    self.sizes/cursor exact regardless of corruption, so checkpoint resume
    never has to reconcile a shifted batch boundary. attempts/failures are
    informational telemetry only (surfaced via coverage()), not a gate --
    persisted across resumes via state_dict/load_state_dict so counts stay
    accurate for the whole run, not just one epoch or one allocation.
    """
    def __init__(self, client, cache, manifest, transform, collate, batch_size=128, seed=0, epochs=3,
                 transform_workers=0):
        self.client, self.cache, self.manifest = client, cache, manifest
        self.transform, self.collate = transform, collate
        self.seed, self.epochs = seed, epochs
        self.transform_workers = transform_workers
        self.sizes = batch_sizes(sum(len(a['members']) for a in manifest['archives']), batch_size)
        self.cursor = {'epoch': 0, 'batch': 0}
        # Live counts advance as samples are produced; _committed is the
        # snapshot state_dict()/checkpointing sees -- frozen between commits,
        # exactly like self.cursor, so a crash between producing and
        # committing a batch doesn't change what a resume sees.
        self.attempts = 0
        self.failures = 0
        self._committed = {'attempts': 0, 'failures': 0}

    def state_dict(self):
        return {**self.cursor, 'fingerprint': self.manifest['fingerprint'], **self._committed}

    def coverage(self):
        completed = self.cursor['epoch']
        return {'completed_epochs': completed,
                'archives_per_completed_epoch': len(self.manifest['archives']),
                'images_per_completed_epoch': sum(self.sizes),
                'images_in_current_epoch': sum(self.sizes[:self.cursor['batch']]),
                'decode_attempts': self.attempts, 'decode_failures': self.failures}

    def load_state_dict(self, state):
        if state['fingerprint'] != self.manifest['fingerprint']:
            raise RuntimeError('Checkpoint manifest mismatch')
        self.cursor = {k: state[k] for k in ('epoch', 'batch')}
        # Older checkpoints (pre-2026-09-22) never recorded these.
        self.attempts = self._committed['attempts'] = state.get('attempts', 0)
        self.failures = self._committed['failures'] = state.get('failures', 0)

    def commit(self, cursor):
        self.cursor = dict(cursor)
        self._committed = {'attempts': self.attempts, 'failures': self.failures}

    def record_failure(self, identity, error):
        self.failures += 1
        print(f'[portal] failed for {identity}: {error!r} -- replacing with a '
              f'blank sample ({self.failures}/{self.attempts} so far)', flush=True)

    @staticmethod
    def decode(raw):
        from PIL import Image, ImageFile

        # The raw portal tier includes a small number of truncated PNGs, and
        # (rarer) an archive member the manifest lists but the archive no
        # longer has (raw=None from __iter__, e.g. a genuine KeyError on
        # extraction). Keep the stream cardinality/cursor stable either way:
        # PIL can recover most truncated files, and both that's-still-
        # unreadable case and the missing-member case become a deterministic
        # blank sample. Dropping either would shift every later batch and
        # invalidate exact checkpoint resume semantics.
        if raw is None:
            return Image.new('RGB', (224, 224))
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(io.BytesIO(raw)) as im:
                return im.convert('RGB')
        except (OSError, ValueError):
            return Image.new('RGB', (224, 224))

    def __iter__(self):
        epoch, batch_start = self.cursor['epoch'], self.cursor['batch']
        if epoch >= self.epochs:
            return
        order = list(self.manifest['archives'])
        random.Random(self.seed + epoch).shuffle(order)
        skip = sum(self.sizes[:batch_start])
        batch, bi = [], batch_start
        wait_started = time.monotonic()
        import multiprocessing
        pool_context = (ProcessPoolExecutor(
            max_workers=self.transform_workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_transform_worker, initargs=(self.transform,))
            if self.transform_workers else nullcontext(None))
        with ThreadPoolExecutor(max_workers=1) as downloads, pool_context as transforms:
            future = None
            for ai, item in enumerate(order):
                names = list(item['members'])
                random.Random(f'{self.seed}:{epoch}:{item["id"]}').shuffle(names)
                if skip >= len(names):
                    skip -= len(names)
                    continue
                pair = order[ai:ai+2]
                paths = self.cache.reserve(pair)
                path = future.result() if future else self.client.download(item, paths[0])
                future = downloads.submit(self.client.download, pair[1], paths[1]) if len(pair) == 2 else None
                names = names[skip:]
                skip = 0
                with archive_reader(path, {**item, '_reserved_bytes': sum(x['size'] for x in pair)}) as read_member:
                    chunk = max(4, self.transform_workers * 4)
                    for start in range(0, len(names), chunk):
                        # Bounded tasks; ordered map preserves exact sample order.
                        tasks = []
                        for n in names[start:start+chunk]:
                            identity = f'{self.seed}:{epoch}:{item["id"]}:{n}'
                            self.attempts += 1
                            try:
                                tasks.append((read_member(n), identity))
                            except (OSError, zipfile.BadZipFile, EOFError, KeyError) as e:
                                self.record_failure(identity, e)
                                tasks.append((None, identity))  # decode() blanks a None raw
                        samples = (transforms.map(_worker_sample, tasks) if transforms else
                                   (_transform_sample(*task, self.transform) for task in tasks))
                        for sample in samples:
                            batch.append(sample)
                            if len(batch) == self.sizes[bi]:
                                bi += 1
                                cursor = {'epoch': epoch + 1, 'batch': 0} if bi == len(self.sizes) else {'epoch': epoch, 'batch': bi}
                                views = self.collate(batch)
                                yield {'views': views, 'cursor': cursor,
                                       'data_wait_seconds': time.monotonic() - wait_started,
                                       'transferred_bytes': self.client.transferred_bytes,
                                       'cache_reserved_peak': self.cache.peak}
                                wait_started = time.monotonic()
                                batch = []
                print(f'[portal] epoch={epoch+1} archive={ai+1}/{len(order)}', flush=True)
                path.unlink()
