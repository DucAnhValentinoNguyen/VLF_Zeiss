# Portal → S3 auth

`cortex.thetavision.nl` is a single-page app behind a data-use agreement; there
is no public API key. `portal_to_s3.py` therefore accepts whatever you can
extract from an authenticated browser session. First match wins.

### Option A — direct / presigned URLs (simplest)
If the portal's download page exposes real links (or you can generate presigned
ones), collect them:
```
/opt/gastronet/urls.txt        # one https://… per line
```
`run_ingest.sh --ingest` picks it up automatically.

### Option B — session cookie + file list
1. Log into the portal in your browser, open DevTools → Network.
2. Trigger one file download; copy the request's `Cookie:` header and the
   per-file endpoint pattern (the recon notes: `/api/file-handling/download/download/<uuid>`).
3. Build `portal.json`:
```json
{
  "base": "https://cortex.thetavision.nl/api/file-handling/download/download",
  "cookie": "sessionid=…; csrftoken=…",
  "file_list": [
    {"name": "gastronet_0001.zip", "url_or_uuid": "b1a2c3d4-…"},
    {"name": "gastronet_0002.zip", "url_or_uuid": "e5f6…"}
  ]
}
```
4. Put it on the box at `/opt/gastronet/portal.json` (via `aws ssm start-session`
   then paste, or `scp` if you set `ingest_key_name`), **or** store the same JSON
   as an AWS Secrets Manager secret and set `portal_secret_arn` in
   `terraform.tfvars` (the instance role can read exactly that one secret).

The session cookie is short-lived — run `--ingest` promptly after capturing it.
It is never written to S3 or logged; only shard names, sizes and sha256s go into
`ingest_log/`.
