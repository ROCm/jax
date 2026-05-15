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

import argparse
import json
from pathlib import Path


def read(path):
    return Path(path).read_text(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--run-code", required=True, type=int)
    parser.add_argument("--run-started-at", required=True)
    parser.add_argument("--run-completed-at", required=True)
    parser.add_argument("--raw", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    benchmark = {
        "target": args.target,
        "workload": args.workload,
        "run_code": args.run_code,
        "run_started_at": args.run_started_at,
        "run_completed_at": args.run_completed_at,
    }

    for raw in args.raw:
        name, path = raw.split("=", 1)
        benchmark[f"{name}_raw"] = read(path)

    Path(args.out).write_text(json.dumps({"benchmark": benchmark}, indent=2) + "\n")


if __name__ == "__main__":
    main()
