#!/usr/bin/env python3

# Collect ROCm CI run metadata and optionally merges additional JSON
# payloads into the final result manifest.
#
# Extra JSON files are merged using dict.update() semantics. Fields in
# extra payloads may override existing manifest fields if names collide.
#
# Benchmark/test-specific payloads should avoid using reserved top-level
# run metadata keys such as:
#  - github_*
#  - run_*
#  - python_version
#  - rocm_

#!/usr/bin/env python3
# Collects ROCm CI run metadata and writes the final result manifest.
#
# Optional --extra JSON payloads are merged into the manifest using
# dict.update() semantics. Extra payloads should avoid reserved run metadata
# keys such as github_*, run_*, python_version, rocm_*, runner, and gpu_count.

import argparse
import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def env(name, default="unknown"):
    return os.environ.get(name, default)


def run(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def sh(command):
    return run(["bash", "-lc", command])


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_json(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def get_response_headers(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.headers


def github_run_started_at():
    repo = env("GITHUB_REPOSITORY", "")
    run_id = env("GITHUB_RUN_ID", "")
    try:
        data = get_json(
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
            headers={"Accept": "application/vnd.github+json"},
        )
        return data.get("run_started_at", "")
    except Exception:
        return ""


def gpu_count(runner):
    match = re.search(r"([0-9]+)gpu", runner or "")
    return int(match.group(1)) if match else None


def image_digest(rocm_tag):
    repo = f"rocm/jax-base-ubu24.{rocm_tag}"
    try:
        token_data = get_json(
            f"https://ghcr.io/token?service=ghcr.io&scope=repository:{repo}:pull"
        )
        token = token_data.get("token", "")
        if not token:
            return ""
        headers = get_response_headers(
            f"https://ghcr.io/v2/{repo}/manifests/latest",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            },
        )
        return headers.get("Docker-Content-Digest", "")
    except Exception:
        return ""


def package_snapshot(python_bin):
    pkgs = run([python_bin, "-m", "pip", "list", "--format=freeze"])
    return "|".join(
        line
        for line in pkgs.splitlines()
        if re.search(r"^(jax|jaxlib)==|pjrt|plugin", line)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--rocm-tag", required=True)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    started = github_run_started_at()
    date = started.split("T", 1)[0] if started else sh("date -u +%F")
    repo = env("GITHUB_REPOSITORY")
    run_id = env("GITHUB_RUN_ID")
    attempt = env("GITHUB_RUN_ATTEMPT")
    workflow = env("GITHUB_WORKFLOW", "")
    manifest = {
        "schema_version": 1,
        "run_key": f"{date}_{run_id}_{attempt}",
        "run_started_at": started,
        "run_completed_at": utc_now(),
        "github_run_url": f"https://github.com/{repo}/actions/runs/{run_id}",
        "github_repository": repo,
        "github_ref_name": env("GITHUB_REF_NAME"),
        "github_ref": env("GITHUB_REF"),
        "github_sha": env("GITHUB_SHA"),
        "github_event_name": env("GITHUB_EVENT_NAME"),
        "github_run_id": run_id,
        "github_run_attempt": attempt,
        "github_run_number": env("GITHUB_RUN_NUMBER"),
        "github_workflow": workflow,
        "is_nightly": "nightly" if "nightly" in workflow.lower() else "continuous",
        "github_job": env("GITHUB_JOB"),
        "python_version": args.python_version,
        "rocm_version": args.rocm_version,
        "rocm_tag": args.rocm_tag,
        "gpu_count": gpu_count(args.runner),
        "runner": args.runner,
        "base_image_name": f"ghcr.io/rocm/jax-base-ubu24.{args.rocm_tag}:latest",
        "base_image_digest": image_digest(args.rocm_tag),
        "jax_packages_raw": package_snapshot(args.python_bin),
        "wheels_sha_raw": sh("sha256sum dist/*.whl 2>/dev/null || true").replace(
            "\n", "|"
        ),
    }

    for extra in args.extra:
        path = Path(extra)
        if path.exists():
            manifest.update(json.loads(path.read_text()))

    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
