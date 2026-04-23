#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

METRIC_FIELD_INDEX = 3
WARMUP_STEPS = 2


def read(path):
    return (
        Path(path).read_text(errors="replace") if path and Path(path).exists() else ""
    )


def parse_metric_values(log_text):
    values = []
    for line in log_text.splitlines():
        if "completed step:" not in line:
            continue
        fields = line.split(",")
        if len(fields) <= METRIC_FIELD_INDEX:
            continue
        try:
            values.append(float(fields[METRIC_FIELD_INDEX].split(":", 1)[1].strip()))
        except Exception:
            pass
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--env-file", default="")
    p.add_argument("--requirements", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--run-code", required=True)
    p.add_argument("--model-run-started-at", required=True)
    p.add_argument("--model-run-completed-at", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    expected = json.loads(Path(args.expected).read_text())
    values = parse_metric_values(read(args.log))

    samples = values[WARMUP_STEPS:]
    observed = statistics.median(samples) if samples else None

    baseline = float(expected["baseline"])
    threshold = float(expected["threshold_percent"])

    distance = None
    cmp_code = 1

    if int(args.run_code) == 0 and observed is not None and baseline != 0:
        raw = ((observed - baseline) / baseline) * 100.0
        distance = abs(raw)
        cmp_code = 0 if raw >= -threshold else 1

    result = {
        "benchmark_schema_version": 1,
        "kind": "benchmark",
        "target": args.target,
        "workload": args.workload,
        "run_code": int(args.run_code),
        "cmp_code": cmp_code,
        "distance_percent": round(distance, 4) if distance is not None else None,
        "model_run_started_at": args.model_run_started_at,
        "model_run_completed_at": args.model_run_completed_at,
        "workload_config_raw": read(args.config),
        "workload_env_raw": read(args.env_file),
        "requirements_raw": read(args.requirements),
        "expected_config_raw": read(args.expected),
    }

    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return cmp_code


if __name__ == "__main__":
    raise SystemExit(main())
