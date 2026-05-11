#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def env(name, default="unknown"):
    return os.environ.get(name, default)


def cmd(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def sh(command):
    return cmd(["bash", "-lc", command])


def github_run_started_at():
    repo = env("GITHUB_REPOSITORY", "")
    run_id = env("GITHUB_RUN_ID", "")
    raw = cmd(
        [
            "curl",
            "-fsSL",
            "-H",
            "Accept: application/vnd.github+json",
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
        ]
    )
    try:
        return json.loads(raw).get("run_started_at", "")
    except Exception:
        return ""


def gpu_count(runner):
    match = re.search(r"([0-9]+)gpu", runner or "")
    return int(match.group(1)) if match else None


def image_digest(rocm_tag):
    repo = f"rocm/jax-base-ubu24.{rocm_tag}"
    token_raw = cmd(
        [
            "curl",
            "-fsSL",
            f"https://ghcr.io/token?service=ghcr.io&scope=repository:{repo}:pull",
        ]
    )
    try:
        token = json.loads(token_raw).get("token", "")
    except Exception:
        return ""

    headers = sh(
        "curl -fsSL -D - "
        f"-H 'Authorization: Bearer {token}' "
        "-H 'Accept: application/vnd.docker.distribution.manifest.v2+json' "
        f"https://ghcr.io/v2/{repo}/manifests/latest "
        "-o /dev/null"
    )
    for line in headers.splitlines():
        if line.lower().startswith("docker-content-digest:"):
            return line.split(":", 1)[1].strip()
    return ""


def package_snapshot():
    python = env("JAXCI_PYTHON", "python3")
    pkgs = cmd([python, "-m", "pip", "list", "--format=freeze"])
    return "|".join(
        line
        for line in pkgs.splitlines()
        if re.search(r"^(jax|jaxlib)==|pjrt|plugin|transformer[-_]engine", line)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = github_run_started_at()
    date = started.split("T", 1)[0] if started else sh("date -u +%F")

    repo = env("GITHUB_REPOSITORY")
    run_id = env("GITHUB_RUN_ID")
    attempt = env("GITHUB_RUN_ATTEMPT")
    workflow = env("GITHUB_WORKFLOW", "")
    runner = env("INPUT_RUNNER")
    rocm_tag = env("INPUT_ROCM_TAG")

    manifest = {
        "schema_version": 1,
        "run_key": f"{date}_{run_id}_{attempt}",
        "run_started_at": started,
        "run_completed_at": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
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
        "python_version": env("INPUT_PYTHON", env("JAXCI_HERMETIC_PYTHON_VERSION")),
        "rocm_version": env("INPUT_ROCM_VERSION"),
        "rocm_tag": rocm_tag,
        "gpu_count": gpu_count(runner),
        "runner": runner,
        "base_image_name": f"ghcr.io/rocm/jax-base-ubu24.{rocm_tag}:latest",
        "base_image_digest": image_digest(rocm_tag),
        "jax_packages_raw": package_snapshot(),
        "wheels_sha_raw": sh("sha256sum dist/*.whl 2>/dev/null || true").replace(
            "\n", "|"
        ),
    }

    for extra in args.extra:
        if extra and Path(extra).exists():
            manifest.update(json.loads(Path(extra).read_text()))

    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
