# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Compares workload benchmark metrics against configured baselines.
#
# Each workload defines:
#   - which metrics are evaluated
#   - how metric values are extracted from logs
#   - whether lower or higher values are better
#   - acceptable regression thresholds
#
# The output contains one comparison result per metric.
import argparse
import json
import statistics
from pathlib import Path
import yaml


def read(path):
    return Path(path).read_text(errors="replace")


def load_workload_config(path, workload):
    config = yaml.safe_load(read(path)) or {}
    if workload not in config:
        raise KeyError(f"Missing workload in baseline config: {workload}")
    return config[workload]


def parse_metric_values(log, metric_config):
    values = []
    pattern = metric_config["log_pattern"]
    field_index = int(metric_config["field_index"])
    warmup_steps = int(metric_config.get("warmup_steps", 0))
    for line in log.splitlines():
        if pattern not in line:
            continue
        try:
            value = line.split(",")[field_index]
            values.append(float(value.split(":", 1)[1].strip()))
        except (IndexError, ValueError):
            continue

    return values[warmup_steps:]


def regression_percent(value, baseline, direction):
    if direction == "lower_is_better":
        regression = ((value - baseline) / baseline) * 100.0
    elif direction == "higher_is_better":
        regression((baseline - value) / baseline) * 100.0
    else:
        raise ValueError(f"Unknown comparison direction: {direction}")

    return abs(regression)


def evaluate_metric(log, name, metric_config):
    values = parse_metric_values(log, metric_config)
    value = statistics.median(values) if values else None

    baseline = float(metric_config["baseline"])
    threshold = float(metric_config["threshold_percent"])
    direction = metric_config["direction"]

    regression = None
    cmp_code = 1
    if value is not None and baseline != 0:
        regression = regression_percent(value, baseline, direction)
        cmp_code = int(regression > threshold)

    return {
        "name": name,
        "value": round(value, 2) if value is not None else None,
        "baseline": baseline,
        "threshold_percent": threshold,
        "direction": direction,
        "regression_percent": (
            round(regression, 4) if regression is not None else None
        ),
        "cmp_code": cmp_code,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log = read(args.log)
    workload = load_workload_config(args.baseline, args.workload)

    metrics = {
        evaluate_metric(log, name, config)
        for name, config in workload["metrics"].items()
    }

    Path(args.out).write_text(
        json.dumps({"benchmark": {"metrics": metrics}}, indent=2) + "\n"
    )

    return max(metric["cmp_code"] for metric in metrics)


if __name__ == "__main__":
    raise SystemExit(main())
