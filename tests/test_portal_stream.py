import io
import shutil
import subprocess
import zipfile

import pytest
from PIL import Image

from vlfz.data.portal import PortalStream, ZipCache, archive_members, archive_read, batch_sizes
from vlfz.data.portal import PortalClient


class RandomizedViews:
    def __call__(self, image):
        import random
        import numpy as np
        import torch
        return (image.getpixel((0, 0))[0], random.random(), float(np.random.rand()),
                float(torch.rand(())))


def test_parallel_transforms_match_serial_and_resume(tmp_path):
    stream = fixture_stream(tmp_path)
    stream.transform = RandomizedViews()
    serial = [batch['views'] for batch in stream]
    stream.transform_workers = 2
    parallel = [batch['views'] for batch in stream]
    assert parallel == serial
    stream.commit({'epoch': 0, 'batch': 1})
    assert [batch['views'] for batch in stream] == serial[1:]


def fixture_stream(tmp_path, epochs=3):
    items = []
    for archive in range(3):
        path = tmp_path / f'source{archive}.zip'
        with zipfile.ZipFile(path, 'w') as z:
            for n in range(3):
                data = io.BytesIO()
                Image.new('RGB', (2, 2), (archive * 3 + n, 0, 0)).save(data, format='PNG')
                z.writestr(f'{n}.png', data.getvalue())
        items.append({'id': archive, 'size': path.stat().st_size,
                      'file_name': path.name, 'members': [f'{n}.png' for n in range(3)]})

    class Client:
        transferred_bytes = 0
        def download(self, item, path):
            if not path.exists():
                shutil.copyfile(tmp_path / item['file_name'], path)
            return path

    cache = ZipCache(tmp_path / 'cache', 4096)
    return PortalStream(Client(), cache, {'fingerprint': 'test', 'archives': items},
                        lambda im: im.getpixel((0, 0))[0], list, 4, epochs=epochs)


def test_three_epochs_cover_every_image(tmp_path):
    stream = fixture_stream(tmp_path)
    orders = []
    for epoch in range(3):
        seen = []
        for batch in stream:
            seen += batch['views']
            stream.commit(batch['cursor'])
        assert sorted(seen) == list(range(9))
        assert stream.cursor == {'epoch': epoch + 1, 'batch': 0}
        orders.append(seen)
    assert orders[0] != orders[1]
    assert list(stream) == []
    assert stream.cache.peak <= 4096


def test_zip_is_opened_once_per_archive(tmp_path, monkeypatch):
    import vlfz.data.portal as portal

    stream = fixture_stream(tmp_path, epochs=1)
    original = portal.zipfile.ZipFile
    opened = 0

    def counted(*args, **kwargs):
        nonlocal opened
        opened += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(portal.zipfile, 'ZipFile', counted)
    for batch in stream:
        stream.commit(batch['cursor'])
    assert opened == 3


def test_single_worker_prefetch_preserves_epoch_cursor(tmp_path):
    import torch

    stream = fixture_stream(tmp_path)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=1, prefetch_factor=1,
        persistent_workers=False,
    )
    for epoch in range(3):
        seen = []
        for batch in loader:
            seen += batch['views']
            stream.commit(batch['cursor'])
        assert sorted(seen) == list(range(9))
        assert stream.cursor == {'epoch': epoch + 1, 'batch': 0}


def test_resume_uses_committed_not_produced_cursor(tmp_path):
    stream = fixture_stream(tmp_path)
    it = iter(stream)
    first = next(it)
    stream.commit(first['cursor'])
    state = stream.state_dict()
    ahead = next(it)
    assert stream.state_dict() == state
    it.close()
    stream.load_state_dict(state)
    assert next(iter(stream))['views'] == ahead['views']


def test_no_singleton_or_dropped_tail():
    assert batch_sizes(9, 4) == [4, 5]
    assert batch_sizes(10, 4) == [4, 4, 2]
    with pytest.raises(ValueError):
        batch_sizes(1, 4)


def test_budget_and_manifest_guard(tmp_path):
    stream = fixture_stream(tmp_path)
    with pytest.raises(RuntimeError):
        stream.cache.reserve([{'id': 8, 'size': 5000}])
    with pytest.raises(RuntimeError):
        stream.load_state_dict({'fingerprint': 'changed'})


class Response:
    def __init__(self, status, data, content_range=None):
        self.status_code = status
        self.headers = {'Content-Range': content_range}
        self.data = data
        self.raw = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, _):
        yield self.data


def client():
    c = object.__new__(PortalClient)
    c.transferred_bytes = 0
    c.url = lambda _: 'https://example.invalid/signed-secret'
    return c


def test_ignored_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    monkeypatch.setattr('vlfz.data.portal.requests.get', lambda *a, **k: Response(200, b'abcd'))
    path = tmp_path / 'partial.zip'
    path.write_bytes(b'ab')
    client().download({'size': 4}, path)
    assert path.read_bytes() == b'abcd'


def test_truncated_download_resumes_exact_range(tmp_path, monkeypatch):
    replies = iter([Response(200, b'ab'), Response(206, b'cd', 'bytes 2-3/4')])
    monkeypatch.setattr('vlfz.data.portal.requests.get', lambda *a, **k: next(replies))
    monkeypatch.setattr('vlfz.data.portal.time.sleep', lambda _: None)
    path = tmp_path / 'partial.zip'
    client().download({'size': 4}, path)
    assert path.read_bytes() == b'abcd'


def test_oversized_download_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr('vlfz.data.portal.requests.get', lambda *a, **k: Response(200, b'abcde'))
    with pytest.raises(RuntimeError, match='exceeded'):
        client().download({'size': 4}, tmp_path / 'partial.zip')


def test_bad_range_fails_without_signed_url(monkeypatch):
    monkeypatch.setattr('vlfz.data.portal.requests.get', lambda *a, **k: Response(200, b'abcd'))
    with pytest.raises(OSError, match='Range unavailable') as exc:
        client().range({'size': 4}, 0, 3)
    assert 'signed-secret' not in str(exc.value)


def test_corrupt_member_is_replaced_to_preserve_coverage(tmp_path):
    # Rewriting the zip with only '0.png' means reading '1.png'/'2.png' (both
    # still declared in the manifest's member list) raises KeyError -- a
    # missing-member case, distinct from 0.png's own now-unreadable content.
    # Both get blanked; cardinality for the epoch (9 images, sizes=[4,5])
    # stays exact, same as test_one_bad_image_is_replaced_not_dropped below.
    stream = fixture_stream(tmp_path)
    with zipfile.ZipFile(tmp_path / 'source0.zip', 'w') as z:
        z.writestr('0.png', b'not an image')
    batches = list(stream)
    assert sum(len(b['views']) for b in batches) == 9


def _corrupt(tmp_path, archive, member):
    """Overwrite one member of one fixture archive with unreadable bytes,
    keeping the other members (and the manifest's declared member list)
    intact -- unlike test_corrupt_member_blocks_coverage, this simulates a
    single truncated image inside an otherwise-valid archive, not a
    manifest/archive mismatch."""
    path = tmp_path / f'source{archive}.zip'
    data = {i.filename: i for i in zipfile.ZipFile(path).infolist()}
    contents = {n: zipfile.ZipFile(path).read(n) for n in data}
    contents[f'{member}.png'] = b'not an image'
    with zipfile.ZipFile(path, 'w') as z:
        for name, raw in contents.items():
            z.writestr(name, raw)


def test_one_bad_image_is_replaced_not_dropped(tmp_path):
    stream = fixture_stream(tmp_path, epochs=1)
    _corrupt(tmp_path, archive=0, member=0)
    batches = list(stream)
    assert sum(len(b['views']) for b in batches) == 9


def test_7z_archives_are_listed_and_read(tmp_path):
    if not shutil.which('7z'):
        pytest.skip('system 7z is not installed')
    data = io.BytesIO()
    Image.new('RGB', (2, 2), (17, 0, 0)).save(data, format='PNG')
    member = tmp_path / 'sample.png'
    member.write_bytes(data.getvalue())
    archive = tmp_path / 'sample.7z'
    subprocess.run(['7z', 'a', str(archive), str(member)], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    item = {'file_name': 'sample.7z'}
    names = archive_members(archive, item)
    assert names == ['sample.png']
    assert archive_read(archive, item, names[0]).startswith(b'\x89PNG')
    from vlfz.data.portal import archive_reader
    with archive_reader(archive, item) as read:
        assert read(names[0]) == data.getvalue()
    assert not list(tmp_path.glob('extract-*'))


@pytest.mark.parametrize('objective', ['lejepa', 'dino'])
def test_lightning_resume_matches_uninterrupted_training(tmp_path, monkeypatch, objective):
    import torch
    import lightning.pytorch as pl
    from types import SimpleNamespace
    from vlfz.cfg import load_cfg
    from vlfz.ssl import pretrain as P

    torch.set_num_threads(1)

    class Backbone(torch.nn.Module):
        embed_dim = 4

        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Module()
            self.trunk.blocks = torch.nn.ModuleList()
            self.trunk.norm = torch.nn.Linear(3, 4)

        def feature(self, x):
            return self.trunk.norm(x)

    monkeypatch.setattr(P, 'build_vit_b16', lambda *a, **k: Backbone())
    monkeypatch.setattr(P, 'provenance', lambda *a, **k: k)
    head = P.DINOHead
    monkeypatch.setattr(P, 'DINOHead', lambda dim, out: head(dim, out, bottleneck=4, hidden=8))
    cfg = load_cfg()
    cfg.ssl.lejepa.proj_dim = 4
    cfg.ssl.lejepa.sigreg_slices = 4
    cfg.ssl.lejepa.sigreg_freqs = 3
    cfg.ssl.ckpt_every_steps = 50
    cfg.ssl.dino.out_dim = 4
    args = SimpleNamespace(init='imagenet', objective=objective, corpus='gastronet',
                           lora=False, limit_steps=0, run_tag='test')

    def run(name, steps, checkpoint=None):
        root = tmp_path / name
        root.mkdir()
        stream = fixture_stream(root)
        stream.transform = lambda im: [torch.tensor(im.getpixel((0, 0)), dtype=torch.float32)/10] * 2
        stream.collate = lambda samples: [torch.stack([s[i] for s in samples]) for i in range(2)]

        class DM(pl.LightningDataModule):
            def __init__(self):
                super().__init__()
                self._dl = torch.utils.data.DataLoader(stream, batch_size=None,
                                                       generator=torch.Generator().manual_seed(0))

            def train_dataloader(self):
                return self._dl

        torch.manual_seed(123)
        model = P.SSLModule(cfg, args, str(root), 6)
        trainer = pl.Trainer(accelerator='cpu', max_epochs=-1, max_steps=steps,
                             logger=False, enable_progress_bar=False, enable_checkpointing=False,
                             plugins=[P.AtomicCheckpointIO()])
        trainer.fit(model, datamodule=DM(), ckpt_path=checkpoint, weights_only=False)
        ckpt = str(root / 'resume.ckpt')
        trainer.save_checkpoint(ckpt)
        return model.state_dict(), ckpt, stream.cursor

    reference, _, _ = run('reference', 6)
    _, checkpoint, cursor = run('interrupted', 1)
    assert cursor == {'epoch': 0, 'batch': 1}
    resumed, _, cursor = run('resumed', 6, checkpoint)
    assert cursor == {'epoch': 3, 'batch': 0}
    for key in reference:
        torch.testing.assert_close(reference[key], resumed[key], rtol=1e-5, atol=1e-6)


def test_failed_checkpoint_replacement_preserves_previous(tmp_path, monkeypatch):
    from vlfz.ssl.pretrain import AtomicCheckpointIO, TorchCheckpointIO
    path = tmp_path / 'last.ckpt'
    path.write_bytes(b'previous-valid-checkpoint')

    def fail(*args, **kwargs):
        raise OSError(122, 'Disk quota exceeded')

    monkeypatch.setattr(TorchCheckpointIO, 'save_checkpoint', fail)
    with pytest.raises(OSError):
        AtomicCheckpointIO().save_checkpoint({}, str(path))
    assert path.read_bytes() == b'previous-valid-checkpoint'
