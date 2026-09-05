"""Drive the GastroNet-5M pipeline from your laptop.

Thin orchestration over Terraform + SSM: it does not reimplement the steps, it
sequences them and waits. If ``prefect`` is installed the steps register as a
flow (nice DAG view / retries); otherwise it runs them straight.

    python pipeline/flow.py up                 # terraform apply
    python pipeline/flow.py run --step ingest  # SSM: portal -> raw/
    python pipeline/flow.py run --step catalog
    python pipeline/flow.py run --step curate
    python pipeline/flow.py run --step dq
    python pipeline/flow.py run --step all
    python pipeline/flow.py status
    python pipeline/flow.py down               # terraform destroy (keeps the bucket)

Stage-in to LRZ is a separate, login-node step: pipeline/stage/stage_in.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

INFRA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infra")

try:
    from prefect import flow, task
except Exception:  # prefect optional
    def task(fn=None, **_):
        return fn if fn else (lambda f: f)

    def flow(fn=None, **_):
        return fn if fn else (lambda f: f)


def _sh(cmd: list[str], cwd: str | None = None) -> str:
    print("+ " + " ".join(cmd))
    return subprocess.check_output(cmd, cwd=cwd, text=True)


def tf_output() -> dict:
    out = _sh(["terraform", "output", "-json"], cwd=INFRA)
    return {k: v["value"] for k, v in json.loads(out).items()}


@task
def up() -> dict:
    _sh(["terraform", "init", "-input=false"], cwd=INFRA)
    _sh(["terraform", "apply", "-auto-approve", "-input=false"], cwd=INFRA)
    o = tf_output()
    print(json.dumps(o, indent=2))
    return o


@task
def run_step(step: str, region: str = "eu-north-1") -> None:
    o = tf_output()
    iid = o["ingest_instance_id"]
    flag = {"ingest": "--ingest", "catalog": "--catalog", "curate": "--curate",
            "dq": "--dq", "all": "--all"}[step]
    cmd = ("source /opt/gastronet/env.sh && "
           f"bash $PIPELINE_SRC/ingest/run_ingest.sh {flag}")
    sent = json.loads(_sh([
        "aws", "ssm", "send-command", "--region", region,
        "--instance-ids", iid, "--document-name", "AWS-RunShellScript",
        "--comment", f"gastronet {step}",
        "--parameters", json.dumps({"commands": [cmd], "executionTimeout": ["172800"]}),
        "--output", "json",
    ]))
    cid = sent["Command"]["CommandId"]
    print(f"SSM command {cid} on {iid}; polling...")
    while True:
        time.sleep(20)
        inv = json.loads(_sh([
            "aws", "ssm", "get-command-invocation", "--region", region,
            "--command-id", cid, "--instance-id", iid, "--output", "json",
        ]))
        st = inv["Status"]
        print(f"  {st}")
        if st in ("Success", "Failed", "Cancelled", "TimedOut"):
            print(inv.get("StandardOutputContent", "")[-4000:])
            if st != "Success":
                print(inv.get("StandardErrorContent", "")[-4000:])
                sys.exit(1)
            return


@task
def status(region: str = "eu-north-1") -> None:
    o = tf_output()
    print(_sh(["aws", "s3", "ls", "--recursive", "--summarize",
               f"s3://{o['bucket']}/", "--human-readable"]))


@task
def down() -> None:
    # keep the data: only tear down the compute + wiring
    _sh(["terraform", "destroy", "-auto-approve", "-input=false",
         "-target=aws_instance.ingest", "-target=aws_launch_template.ingest"], cwd=INFRA)


@flow(name="gastronet5m-pipeline")
def all_steps(region: str = "eu-north-1") -> None:
    up()
    for s in ("ingest", "catalog", "curate", "dq"):
        run_step(s, region)
    status(region)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up")
    r = sub.add_parser("run")
    r.add_argument("--step", required=True, choices=["ingest", "catalog", "curate", "dq", "all"])
    r.add_argument("--region", default="eu-north-1")
    sub.add_parser("status").add_argument("--region", default="eu-north-1")
    sub.add_parser("down")
    sub.add_parser("flow").add_argument("--region", default="eu-north-1")
    a = ap.parse_args()

    if a.cmd == "up":
        up()
    elif a.cmd == "run":
        all_steps(a.region) if a.step == "all" else run_step(a.step, a.region)
    elif a.cmd == "status":
        status(a.region)
    elif a.cmd == "down":
        down()
    elif a.cmd == "flow":
        all_steps(a.region)


if __name__ == "__main__":
    main()
