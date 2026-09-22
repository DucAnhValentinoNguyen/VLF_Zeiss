# Full-corpus portal training

Submit from the repository with `bash lrz/submit_fullstream.sh`. The default
objective is now DINO; set `STREAM_OBJECTIVES=lejepa` for the independent LeJEPA
job. Each objective has an exclusive lock, and LeJEPA uses `zip_cache_lejepa`.
Evaluation holds a shared lock to protect feature caches and reports.
The original combined workflow runs two
200-step pilots followed by DINO then LeJEPA, ImageNet initialization, three
complete raw-image epochs, one GPU. Evaluation follows each completed objective.
The tag is `s506_raw_e3`; pilot tags have an additional `_pilot` suffix.

All new writable paths are under the real home directory
`~/vlf_zeiss_fullstream`. The ZIP cache reserves at most 16 GiB, including
partially downloaded files, and holds an exclusive process lock. The GPU job
requires room below the home soft quota for growth to the 16 GiB cache limit
plus 8 GiB for checkpoint operations at startup. Existing project data is
read only for downstream evaluation. No permanent ingest is required.

Authentication uses `PORTAL_JSON` (default `~/.gastronet_portal/portal.json`).
An optional private `CORTEX_ACCESS_URL` enables session renewal. Do not put
credentials, access links, or signed download URLs into version control.

The manifest is indexed before training. Every archive must be readable;
corrupt images/archives stop execution rather than silently reducing coverage.
Each epoch shuffles archives and members deterministically. Training commits
the cursor only after the optimizer step; checkpoint replay uses the saved
cursor and deterministic sample augmentation. Epoch tails are retained, with
a singleton merged into the preceding batch.

Only exit 75 following a wall-time checkpoint triggers automatic resubmission,
with at most 100 resumptions and 24-hour allocations. Other failures stop for inspection. Logs live in
`lrz/logs/fullstream_*.out` and the output directory. Refresh expired credentials
and resubmit manually after diagnosing a stopped run; receipts skip completed
phases. Checkpoints are atomically replaced. A pilot CUDA OOM retries once
with gradient checkpointing enabled.

Three epochs for each objective transfer about 11.25 TB, plus pilots/retries.
Full-corpus raw preprocessing differs from the historical 60-ZIP resized,
JPEG-reencoded, pHash-deduplicated tier. Score differences cannot be attributed
solely to corpus size. LeJEPA's earlier regression remains an evaluation question.

Acceptance requires both training cursors to show three complete epochs over
the frozen 506-archive manifest, successful tagged evaluations, and published
W&B model artifacts. Submission or a partial checkpoint is not completion.

## Training optimization rollout (2026-09-18)

- DINO continuation: job 5796919; LeJEPA: job 5796922. Both pending on priority
  at submission verification. Superseded pending job 5796510 was cancelled.
- ZIPs stay open for an entire archive. 7z archives are extracted once into a
  temporary directory after checking declared expansion plus both reserved
  archives against 16 GiB. Oversized combinations stop for inspection.
- Four spawned CPU processes decode and augment images using sample-specific
  seeds. Bounded, ordered work preserves sample order and resume cursors.
  DataLoader workers are zero because daemon workers cannot create this pool.
- Every 100 new steps prints `[throughput]` with elapsed seconds and measured
  steps/second. Use this for ETA; a resumed progress bar can misstate speed.
- Parallel/serial random output and resumed output match in tests; existing
  Lightning checkpoint-equivalence checks pass for both objectives (19 tests).
- GPU throughput improvement remains unmeasured until the queued jobs start.
