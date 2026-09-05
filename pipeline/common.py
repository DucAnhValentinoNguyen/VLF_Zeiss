"""Shared helpers for the GastroNet-5M data pipeline (S3 + shard grouping).

Deliberately dependency-light: boto3 + stdlib. Nothing here imports torch or
`vlfz` — the pipeline runs on a bare EC2 box and on the LRZ login node alike.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterator

import boto3
from botocore.config import Config

_BOTO = Config(retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=32)


def s3():
    return boto3.client("s3", config=_BOTO)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    m = re.match(r"^s3://([^/]+)/?(.*)$", uri)
    if not m:
        raise ValueError(f"not an s3:// uri: {uri}")
    return m.group(1), m.group(2)


def list_keys(bucket: str, prefix: str, suffix: str = "") -> Iterator[str]:
    paginator = s3().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):
                continue
            if suffix and not k.lower().endswith(suffix):
                continue
            yield k


def key_exists(bucket: str, key: str) -> dict | None:
    try:
        return s3().head_object(Bucket=bucket, Key=key)
    except s3().exceptions.ClientError:
        return None


def upload_file(bucket: str, key: str, path: str, extra: dict | None = None) -> None:
    s3().upload_file(path, bucket, key, ExtraArgs=extra or {})


def download_file(bucket: str, key: str, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    s3().download_file(bucket, key, path)
    return path


def sha256_file(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- shard grouping
# The portal ships ~506 zip shards. We bucket them into `shard_group`s so the
# Parquet catalog is Hive-partitioned and Athena scans stay small, and so the
# curation job has natural parallel work units.
SHARDS_PER_GROUP = 16


@dataclass(frozen=True)
class ShardRef:
    key: str          # s3 key of the zip, e.g. "raw/gastronet_0007.zip"
    name: str         # basename without extension
    index: int        # numeric shard index parsed from the name (fallback: enumerate order)
    group: str        # partition value, e.g. "g000"


def _shard_index(name: str, fallback: int) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else fallback


def shard_refs(bucket: str, raw_prefix: str) -> list[ShardRef]:
    keys = sorted(list_keys(bucket, raw_prefix, ".zip"))
    refs = []
    for i, k in enumerate(keys):
        name = os.path.splitext(os.path.basename(k))[0]
        idx = _shard_index(name, i)
        refs.append(ShardRef(key=k, name=name, index=idx, group=f"g{idx // SHARDS_PER_GROUP:03d}"))
    return refs


def groups(refs: list[ShardRef]) -> dict[str, list[ShardRef]]:
    out: dict[str, list[ShardRef]] = {}
    for r in refs:
        out.setdefault(r.group, []).append(r)
    return out
